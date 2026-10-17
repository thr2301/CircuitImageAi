from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from services.user_repo import UserRepository

router = APIRouter()
user_repo = UserRepository()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
# Display the login page.
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
# Authenticate a user and redirect to the dashboard on success.
def login(username: str = Form(...), password: str = Form(...)):
    if user_repo.authenticate(username, password):
        r = RedirectResponse("/dashboard", 303)
        r.set_cookie("logged_in", "true", httponly=True)
        r.set_cookie("username", username, httponly=True)
        return r
    return RedirectResponse("/", 303)

@router.get("/logout")
# Clear the login cookies and redirect to the login page.
def logout_route():
    r = RedirectResponse("/", 303)
    r.delete_cookie("logged_in")
    r.delete_cookie("username")
    return r

@router.get("/register", response_class=HTMLResponse)
# Display the registration page.
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.post("/register")
# Create a user account and redirect with the registration result.
def register(username: str = Form(...), password: str = Form(...), email: str = Form(...), phone: str = Form(None)):
    success = user_repo.add_user(username, password, email, phone)
    if not success:
        return RedirectResponse("/register?error=exists", 303)
    return RedirectResponse("/?registered=1", 303)

@router.get("/forgot-password", response_class=HTMLResponse)
# Display the password recovery page.
def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})

@router.post("/forgot-password")
# Update a user's password using their email address.
def forgot_password_submit(email: str = Form(...), new_password: str = Form(...)):
    if user_repo.update_password_by_email(email, new_password):
        return RedirectResponse("/?message=PasswordResetSuccess", 303)
    return RedirectResponse("/forgot-password?error=EmailNotFound", 303)

@router.get("/dashboard", response_class=HTMLResponse)
# Show the dashboard to an authenticated user.
def dashboard(request: Request):
    if request.cookies.get("logged_in") != "true":
        return RedirectResponse("/", 303)
    username = request.cookies.get("username")
    current_user = user_repo.get_user_by_username(username)
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user})

@router.post("/update_profile")
# Update the logged-in user's profile details.
def update_profile(request: Request, email: str = Form(...), phone: str = Form(None), new_password: str = Form(None)):
    username = request.cookies.get("username")
    if not username:
        return RedirectResponse("/", 303)
    user_repo.update_user_profile(username, email, phone, new_password)
    return RedirectResponse("/dashboard?message=ProfileUpdated", 303)