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
    analysis: str
    status: str = "success"


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
    then uses NVIDIA's DeepSeek R1 model to generate an analysis report.
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

        # Run the LangChain agent
        analysis = analyze_weekly_trades(
            user_id=request.user_id,
            start_date=request.start_date,
            end_date=request.end_date,
            query=request.query,
        )

        return WeeklyAnalysisResponse(analysis=analysis)

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
