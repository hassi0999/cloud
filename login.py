from fastapi import FastAPI, Form, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from db import Base, SessionLocal, engine
from model import JobSubmission, enrty
from typing import Any, Dict
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
from pydantic import EmailStr
import requests
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi import Query
app = FastAPI(openapi_url="/api/v1/openapi.json")

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8008"],  # Update with your client's origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Include OPTIONS method for preflight requests
    allow_headers=["*"],
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="My Api ",
        version="1.0.0",
        description="This is the API documentation for your application.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


# Define a Pydantic model to represent the response from the external AI service
class AIResponse(BaseModel):
    status_code: int
    response_json: Dict[str, Any]


# app = FastAPI()
PRICING_TIERS = {"basic": 5, "standard": 250, "premium": 500}
templates = Jinja2Templates(directory="templates")
Base.metadata.create_all(engine)
SECRET_KEY = " Khushab "
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
app.mount(
    "/static", StaticFiles(directory="D:/PYTHONAPI/project/templates"), name="static"
)

logging.basicConfig(filename='server.log', level=logging.DEBUG)

# Route to serve the index.html file from the root path
@app.get("/client", response_class=HTMLResponse)
async def get_index():
    return FileResponse(r"D:\fastapi\fastapi\templates\client.html")


@app.get("/pricing", response_class=HTMLResponse)
async def get_pricing():
    return FileResponse("D:/PYTHONAPI/project/templates/pricing.html")
# Generate JWT token


def create_access_token(email: str):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": email, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Route to display the login form
@app.get("/", response_class=HTMLResponse)
async def show_login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Route to handle user login
@app.post("/login/")
async def login(
    request: Request, email: EmailStr = Form(...), password: str = Form(...)
):

    # Establish a database session
    with SessionLocal() as db:
        # Query the database for the user
        user = db.query(enrty).filter(enrty.email == email).first()

        # Check if user exists and password is correct
        if not user or not verify_password(password, user.password):
            raise HTTPException(status_code=401, detail="Incorrect email or password")

        # Generate access token
        token = create_access_token(email)

    # Return the access token
    return JSONResponse(content={"access_token": token, "token_type": "bearer"})


# Route to handle user signup
@app.post("/signup/")
async def signup(
    email: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(...),
    pricing_tier: str = Form(...),
):
    credits = PRICING_TIERS[pricing_tier]
    hashed_password = pwd_context.hash(password)
    new_user = enrty(
        email=email, password=hashed_password, full_name=full_name, credits=credits
    )
    db = SessionLocal()
    db.add(new_user)
    db.commit()
    db.close()
    access_token = create_access_token(email)
    return {"message": "Sign up successfuly"}
    return {"message": "Sign up successful", "access_token": access_token}

def deduct_credits(user_id, amount_to_deduct):
    user = SessionLocal().query(enrty).get(user_id)
    if user:
        if user.credits >= amount_to_deduct:
            user.credits -= amount_to_deduct
            SessionLocal().commit()
            return True
    return False


def has_sufficient_credits(user_id, required_credits):
    user = SessionLocal().query(enrty).get(user_id)
    return user.credits >= required_credits if user else False


def add_credits(user_id, amount_to_add):
    user = SessionLocal().query(enrty).get(user_id)
    if user:
        user.credits += amount_to_add
        SessionLocal().commit()


def get_token(authorization: str = Header(...)):
    if authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1]
        return token
    raise HTTPException(status_code=401, detail="Invalid authorization header")


def is_token_valid(token: str):
    try:
        # Decode the token to extract expiration time
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        expiration = payload.get("exp")

        # Check if the token has expired
        if expiration is None or datetime.utcnow() > datetime.fromtimestamp(expiration):
            return False, "Token expired"
        return True, "Token valid"
    except JWTError:
        return False, "Invalid token"


# Define a queue to hold pending job submissions
job_queue = []


def get_user_by_email(email: str, db: Session = Depends(SessionLocal)):
    user = db.query(enrty).filter(enrty.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# Route to submit a job
@app.post("/submit_job/")
async def submit_job(
    request: Request,
    email: str = Form(...),  # Assume email is provided in the form
    job_title: str = Form(...),
    description: str = Form(...),
    deadline: str = Form(...),  # Change parameter name to "deadline"
    priority: str = Form(...),
    token: str = Depends(get_token),  # Add a dependency to get the access token
):
    # Check if access token is valid
    token_valid, token_message = is_token_valid(token)
    if not token_valid:
        if token_message == "Token expired":
            raise HTTPException(
                status_code=401, detail="Token expired. Please login again. "
            )
            
        else:
            raise HTTPException(status_code=403, detail="Invalid token")
        
    db = SessionLocal()

    # Get user by email
    user = get_user_by_email(email, db)

    # Check if user has sufficient credits
    if user.credits <= 0:
        db.close()
        return JSONResponse(status_code=403,content={"message": "Insufficient credits.your job is in qeue."},)

    # Deduct credits
    user.credits -= 1

    # Convert deadline string to datetime object
    deadline_dt = datetime.fromisoformat(deadline)

    # Create job submission
    job_submission = JobSubmission(
        user_id=user.id,
        job_title=job_title,
        description=description,
        deadline=deadline_dt,  # Use the converted deadline datetime object
        priority=priority,
    )
    db.add(job_submission)
    db.commit()
    db.close()
    return {"message": "Job submitted successfully"}


# External API configuration
# API_URL = "https://api.openai.com/v1/chat/completions"
# API_KEY = "sk-proj-yH78NFT4cQ20Ig4pae8FT3BlbkFJxTue5mfiQJBAXiIVu9rW"  # Your OpenAI API key
API_URL = "https://api.ai21.com/studio/v1/j2-ultra/chat"    
API_KEY =  "uv8OpR04ePSzhVZ9xo3tYrvYUbHtX0nH"

# Define a Pydantic model to represent the request body for the external API
class AIRequest(BaseModel):
    prompt: str

# Define a Pydantic model to represent the response from the external AI service
class AIResponse(BaseModel):
    status_code: int
    response_json: Dict[str, Any]


# Function to make the API call to the external AI service
async def call_external_ai_service(request: Request):
    async def call_external_ai_service(request: Request):
        response = await client.post(f"{API_URL}", headers=headers, json={"prompt": request.prompt})

# Endpoint to handle the API request to the external AI service
@app.post("/complete")
async def complete_text(request: AIRequest, response: AIResponse = Depends(call_external_ai_service)):
    logging.debug("Received POST request to /complete/ endpoint")  # Log request received
    if response.status_code == 200:
        # Handle successful response from the external AI service
        ai_result = response.response_json
        return {"result": ai_result}
    elif response.status_code == 401:
        # Unauthorized error
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")
    elif response.status_code == 422:
        # Unprocessable Entity error
        raise HTTPException(
            status_code=422, detail="Unprocessable Entity: Invalid request"
        )
    elif response.status_code == 500:
        raise HTTPException(status_code=500,detail= "Error from external AI service")    
    else:
        # Other errors
        raise HTTPException(
            status_code=response.status_code, detail="Error from external AI service"
        )


def process_job(user_id, job_details, credits_required):
    if not deduct_credits(user_id, credits_required):
        raise HTTPException(status_code=500, detail="Failed to deduct credits")

    # Process the job here

    

    # Example: Print job details
    print(f"Job submitted by user {user_id}: {job_details}")

    # After processing the job, you might want to check the queue for pending jobs
    process_pending_jobs()


def process_pending_jobs():
    while job_queue:
        user_id, job_details, credits_required = job_queue.pop(0)
        if has_sufficient_credits(user_id, credits_required):
            # Process the pending job if user now has sufficient credits
            process_job(user_id, job_details, credits_required)
        else:
            # Put the job back in the queue if user still does not have sufficient credits
            job_queue.append((user_id, job_details, credits_required))


# Route to display the signup form
@app.get("/signup/", response_class=HTMLResponse)
async def show_signup_form(request: Request):
    return templates.TemplateResponse("SignUp.html", {"request": request})


@app.get("/job/", response_class=HTMLResponse)
async def show_job_form(request: Request):
    return templates.TemplateResponse("job.html", {"request": request})


@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_endpoint():
    return app.openapi()


@app.get("/openapi.yaml", include_in_schema=False)
async def get_openapi_yaml_endpoint():
    return JSONResponse(content=app.openapi(), media_type="application/x-yaml")

@app.get("/pdf")
async def get_df():
    pdf_path = r"D:\PYTHONAPI\project\APIDesignJustification.pdf"
    return FileResponse(pdf_path)


@app.post("/chat/")
async def chat_with_ai(user_input: str = Query(...)):
    url = "https://api.ai21.com/studio/v1/j2-ultra/chat"
    api_key = "pg3AZXVOUbUJSObwCq5m7dIQ07u09xZc"  # Replace with your actual API key
    # Prepare the payload with user input
    payload = {
        "numResults": 1,
        "temperature": 0.7,
        "messages": [
            {
                "text": user_input,  # Use user input as the message text
                "role": "user"
            }
        ],
        "system": "You are an AI assistant for business research. Your responses should be informative and concise."
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"  # Correctly format the API key
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail="Error communicating with the external AI service")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Modify the buy_credits function to use email_check
@app.post("/buy_credits/")
async def buy_credits(
    email: str = Form(...),
    pricing_tier: str = Form(...),
    db: Session = Depends(get_db)
):
    # Fetch the user from the database
    user = db.query(enrty).filter(enrty.email == email).first()

    # Update user's credits based on the selected pricing tier
    if pricing_tier == "basic":
        user.credits += 50
    elif pricing_tier == "standard":
        user.credits += 100
    elif pricing_tier == "premium":
        user.credits += 200

    # Commit the changes to the database
    db.commit()

    # Return a success message
    
    return {"message": "Credits added successfully", "new_credits": user.credits}
