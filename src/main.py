"""
FastAPI server for Trade Tracker AI Analysis.
Provides the /weekly-analysis endpoint for the React frontend.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import analyze_weekly_trades

app = FastAPI(
    title="Trade Tracker AI Agent",
    description="LangChain-based AI agent for weekly trade analysis",
    version="1.0.0",
)

# CORS — allow React frontend (local dev + production Vercel URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://trading-tracker-two.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WeeklyAnalysisRequest(BaseModel):
    user_id: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    query: str = "Can you tell me last week's trade analysis?"


class WeeklyAnalysisResponse(BaseModel):
    analysis: str | None = None
    status: str = "success"  # "success" | "no_trades"


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Trade Tracker AI Agent is running",
        "endpoints": {
            "/weekly-analysis": "POST - Generate weekly trade analysis",
        },
    }


@app.post("/weekly-analysis", response_model=WeeklyAnalysisResponse)
async def weekly_analysis(request: WeeklyAnalysisRequest):
    """
    Generate AI-powered weekly trade analysis.

    Fetches trade data from Firebase for the given user and date range,
    then uses NVIDIA's Nemotron model to generate an analysis report.
    Returns status="no_trades" immediately (without calling the LLM) when
    there are no trades in the selected date range.
    """
    try:
        # Validate date format
        from datetime import datetime

        try:
            datetime.strptime(request.start_date, "%Y-%m-%d")
            datetime.strptime(request.end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Dates must be in YYYY-MM-DD format",
            )

        if not request.user_id.strip():
            raise HTTPException(
                status_code=400,
                detail="user_id is required",
            )

        # Run the agent (returns dict: {has_trades, analysis})
        result = analyze_weekly_trades(
            user_id=request.user_id,
            start_date=request.start_date,
            end_date=request.end_date,
            query=request.query,
        )

        # No trades found — return immediately without LLM call
        if not result["has_trades"]:
            return WeeklyAnalysisResponse(analysis=None, status="no_trades")

        return WeeklyAnalysisResponse(analysis=result["analysis"], status="success")

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
