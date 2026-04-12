"""
LangChain AI Agent for Trade Tracker weekly analysis.
Uses NVIDIA's DeepSeek R1 model via LangChain to analyze trade data.
"""
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from firebase_utils import fetch_trades
from dotenv import load_dotenv
import os
import logging
import time

load_dotenv()

logger = logging.getLogger(__name__)

# List of reliable models to try in sequence (Robust Fallback Mechanism)
MODEL_NAMES = [
    "meta/llama-3.3-70b-instruct",
    "deepseek-ai/deepseek-v3.2",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "google/gemma-3-27b-it"
]

# Prompt template tailored to your Trade Tracker data structure
analysis_prompt = PromptTemplate(
    input_variables=["query", "trade_data", "trade_summary"],
    template="""You are an expert trading analyst assistant. Your task is to analyze the user's trade data and provide a comprehensive weekly analysis report.

User Query: {query}

=== TRADE SUMMARY ===
{trade_summary}

=== DETAILED TRADE DATA ===
{trade_data}

=== INSTRUCTIONS ===
Provide a well-structured weekly analysis report with the following sections:

1. **📊 Weekly Overview**: Summarize the trading activity for the week (total trades, intraday vs delivery breakdown).

2. **💰 Profit/Loss Summary**: 
   - Total net P/L
   - Best performing trade
   - Worst performing trade
   - Win rate (profitable trades / total closed trades)

3. **📈 Patterns & Trends**:
   - Which stocks were traded most frequently
   - Any patterns in winning vs losing trades
   - Common mistakes identified from the trader's self-reported mistakes

4. **🎯 Actionable Recommendations**:
   - Specific, practical advice based on the data
   - Areas for improvement based on the mistakes analysis
   - Lessons the trader should focus on

5. **⚠️ Risk Assessment**:
   - Evaluate position sizing and risk management
   - Flag any concerning patterns

Keep the analysis data-driven and actionable. Use the actual numbers from the trade data. Format currency in ₹ (Indian Rupees).

Response:
""",
)


def calculate_trade_pnl(trade: dict) -> dict:
    """Calculate profit/loss for a single trade, matching the frontend logic."""
    buy_price = float(trade.get("buyPrice", 0))
    sell_price = float(trade.get("sellPrice", 0))
    quantity = int(trade.get("quantity", 0))
    brokerage = float(trade.get("brokerage", 0))
    taxes = float(trade.get("taxes", 0))
    trade_type = trade.get("tradeType", "intraday")

    # For delivery trades without sell price, mark as holding
    if trade_type == "delivery" and sell_price == 0:
        return {
            "grossProfit": None,
            "netProfit": None,
            "totalCharges": round(brokerage + taxes, 2),
            "status": "holding",
        }

    gross_profit = (sell_price - buy_price) * quantity
    total_charges = brokerage + taxes
    net_profit = gross_profit - total_charges

    return {
        "grossProfit": round(gross_profit, 2),
        "netProfit": round(net_profit, 2),
        "totalCharges": round(total_charges, 2),
        "status": "closed",
    }


def generate_trade_summary(trades: list[dict]) -> str:
    """Generate a statistical summary from trade data."""
    if not trades:
        return "No trades found for the specified period."

    total_trades = len(trades)
    intraday_count = sum(1 for t in trades if t.get("tradeType") == "intraday")
    delivery_count = total_trades - intraday_count

    profit_trades = 0
    loss_trades = 0
    total_profit = 0.0
    total_loss = 0.0
    holding_count = 0
    best_trade = None
    worst_trade = None
    best_pnl = float("-inf")
    worst_pnl = float("inf")

    # Collect all mistakes
    all_mistakes = {}

    for trade in trades:
        pnl = calculate_trade_pnl(trade)

        if pnl["status"] == "holding":
            holding_count += 1
            continue

        net = pnl["netProfit"]
        if net > 0:
            profit_trades += 1
            total_profit += net
        else:
            loss_trades += 1
            total_loss += abs(net)

        if net > best_pnl:
            best_pnl = net
            best_trade = trade
        if net < worst_pnl:
            worst_pnl = net
            worst_trade = trade

        # Track mistakes
        for mistake in trade.get("mistakesMade", []):
            all_mistakes[mistake] = all_mistakes.get(mistake, 0) + 1

    closed_trades = profit_trades + loss_trades
    win_rate = (profit_trades / closed_trades * 100) if closed_trades > 0 else 0
    net_pnl = total_profit - total_loss

    summary_lines = [
        f"Total trades: {total_trades}",
        f"Intraday: {intraday_count} | Delivery: {delivery_count}",
        f"Closed trades: {closed_trades} | Holding: {holding_count}",
        f"Profitable trades: {profit_trades} | Loss trades: {loss_trades}",
        f"Win rate: {win_rate:.1f}%",
        f"Total profit: ₹{total_profit:.2f}",
        f"Total loss: ₹{total_loss:.2f}",
        f"Net P/L: ₹{net_pnl:.2f}",
    ]

    if best_trade:
        summary_lines.append(
            f"Best trade: {best_trade['stockName']} (₹{best_pnl:.2f})"
        )
    if worst_trade:
        summary_lines.append(
            f"Worst trade: {worst_trade['stockName']} (₹{worst_pnl:.2f})"
        )

    if all_mistakes:
        sorted_mistakes = sorted(all_mistakes.items(), key=lambda x: x[1], reverse=True)
        mistakes_str = ", ".join(f"{m} ({c}x)" for m, c in sorted_mistakes)
        summary_lines.append(f"Common mistakes: {mistakes_str}")

    return "\n".join(summary_lines)


def format_trade_data(trades: list[dict]) -> str:
    """Format trade data into a readable string for the LLM."""
    if not trades:
        return "No trade data available."

    lines = []
    for i, trade in enumerate(trades, 1):
        pnl = calculate_trade_pnl(trade)
        pnl_str = f"₹{pnl['netProfit']:.2f}" if pnl["netProfit"] is not None else "Holding"

        line = (
            f"Trade #{i}: {trade['stockName']} ({trade['tradeType'].upper()}) | "
            f"Buy: ₹{trade['buyPrice']} → Sell: ₹{trade['sellPrice']} | "
            f"Qty: {trade['quantity']} | "
            f"Charges: ₹{float(trade.get('brokerage', 0)) + float(trade.get('taxes', 0)):.2f} | "
            f"Net P/L: {pnl_str} | "
            f"Date: {trade.get('buyDate', 'N/A')}"
        )

        if trade.get("mistakesMade"):
            line += f" | Mistakes: {', '.join(trade['mistakesMade'])}"
        if trade.get("lessonsLearned"):
            line += f" | Lesson: {trade['lessonsLearned'][:100]}"

        lines.append(line)

    return "\n".join(lines)


def analyze_weekly_trades(
    user_id: str, start_date: str, end_date: str, query: str
) -> str:
    """
    Main function: fetch trades from Firebase and generate AI analysis.

    Args:
        user_id: Firebase Auth UID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        query: User's question/prompt

    Returns:
        AI-generated analysis string
    """
    # 1. Fetch trades from Firebase
    trades = fetch_trades(user_id, start_date, end_date)

    if not trades:
        return (
            f"No trades found for the period {start_date} to {end_date}. "
            "Please make sure you have trades recorded in this date range."
        )

    # 2. Prepare data for the LLM
    trade_data_str = format_trade_data(trades)
    trade_summary_str = generate_trade_summary(trades)

    # 3. Build the prompt
    formatted_prompt = analysis_prompt.format(
        query=query,
        trade_data=trade_data_str,
        trade_summary=trade_summary_str,
    )

    # 4. Call the NVIDIA LLM via LangChain using fallback logic
    response = invoke_with_fallback(formatted_prompt)

    return response.content


def invoke_with_fallback(prompt: str):
    """Try invoking multiple models in sequence until one succeeds."""
    last_error = None
    
    for i, model_name in enumerate(MODEL_NAMES, 1):
        try:
            logger.info(f"Attempt {i}/{len(MODEL_NAMES)} - Trying AI model: {model_name}...")
            llm = ChatNVIDIA(
                model=model_name,
                api_key=os.getenv("NVIDIA_API_KEY"),
                temperature=0.6,
                top_p=0.7,
                max_tokens=4096,
            )
            return llm.invoke(prompt)
        except Exception as e:
            last_error = e
            logger.warning(f"Model '{model_name}' failed: {e}")
            if i < len(MODEL_NAMES):
                logger.info("Switching to next fallback model...")
            
    logger.error("All fallback models failed. NVIDIA servers might be temporarily down or keys are invalid.")
    raise Exception(
        f"Analysis failed: Tried {len(MODEL_NAMES)} different AI models but all failed. Last error: {last_error}"
    )