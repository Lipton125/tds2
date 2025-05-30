from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import httpx
import os
from typing import Dict, Any

app = FastAPI()

# Set up OpenAI's API Key (replace this with your actual key)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define the function signatures
FUNCTIONS = [
    {
        "name": "get_ticket_status",
        "description": "Get the status of a ticket based on the ticket ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "integer", "description": "The ID of the support ticket."}
            },
            "required": ["ticket_id"],
            "additionalProperties": False
        },
    },
    {
        "name": "schedule_meeting",
        "description": "Schedule a meeting with a specific date, time, and room.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "The date of the meeting (YYYY-MM-DD)."},
                "time": {"type": "string", "description": "The time of the meeting (HH:MM)."},
                "meeting_room": {"type": "string", "description": "The meeting room."}
            },
            "required": ["date", "time", "meeting_room"],
            "additionalProperties": False
        },
    },
    {
        "name": "get_expense_balance",
        "description": "Get the expense balance for an employee based on the employee ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "integer", "description": "The ID of the employee."}
            },
            "required": ["employee_id"],
            "additionalProperties": False
        },
    },
    {
        "name": "calculate_performance_bonus",
        "description": "Calculate the performance bonus for an employee in a specific year.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "integer", "description": "The ID of the employee."},
                "current_year": {"type": "integer", "description": "The year for the bonus calculation."}
            },
            "required": ["employee_id", "current_year"],
            "additionalProperties": False
        },
    },
    {
        "name": "report_office_issue",
        "description": "Report an office issue to a specific department based on an issue code.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_code": {"type": "integer", "description": "The code of the office issue."},
                "department": {"type": "string", "description": "The department handling the issue."}
            },
            "required": ["issue_code", "department"],
            "additionalProperties": False
        },
    }
]

# Helper function to query OpenAI API
async def query_openai(user_input: str, functions: list) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": user_input}],
                "functions": functions,
                "tool_choice": "auto",  # This automatically lets OpenAI choose which tool to use
            },
        )
        return response.json()

@app.get("/execute")
async def execute(q: str):
    try:
        # Send the query to OpenAI and get the function call suggestion
        response = await query_openai(q, FUNCTIONS)

        # Parse the response from OpenAI
        function_name = response["choices"][0]["message"].get("function_call", {}).get("name")
        arguments = response["choices"][0]["message"].get("function_call", {}).get("arguments", {})

        if not function_name:
            raise HTTPException(status_code=400, detail="No valid function call found.")

        # Ensure arguments are in the correct format
        arguments = {k: v for k, v in arguments.items()}  # Ensure it's in dictionary format

        # Return the function name and arguments as a response
        return JSONResponse(content={
            "name": function_name,
            "arguments": arguments
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
