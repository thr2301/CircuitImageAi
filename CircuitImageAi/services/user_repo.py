import json
import os
from typing import List, Dict, Optional
from passlib.hash import bcrypt
from core.utils import USERS_FILE, ensure_file_exists

class UserRepository:
    # Initialize the user repository and ensure its storage file exists.
    def __init__(self):
        ensure_file_exists(USERS_FILE, [])

    # Load and return all users from storage.
    def load_users(self) -> List[Dict]:
        ensure_file_exists(USERS_FILE, [])
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    # Save all users to storage.
    def save_users(self, users: List[Dict]):
        tmp = USERS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        os.replace(tmp, USERS_FILE)

    # Verify a user's password against the stored password hash.
    def authenticate(self, username: str, password: str) -> bool:
        for u in self.load_users():
            if u.get("username") == username:
                try:
                    return bcrypt.verify(password, u.get("password", ""))
                except Exception:
                    return False
        return False

    # Create and store a new user with a hashed password.
    def add_user(self, username: str, password: str, email: str, phone: Optional[str] = None) -> bool:
        users = self.load_users()
        if any(u["username"] == username for u in users):
            return False
        users.append({
            "id": len(users) + 1,
            "username": username,
            "password": bcrypt.hash(password),
            "email": email,
            "phone": phone
        })
        self.save_users(users)
        return True

    # Update a user's email, phone number, and optionally password.
    def update_user_profile(self, username: str, email: str, phone: Optional[str], new_password: Optional[str]):
        users = self.load_users()
        for u in users:
            if u["username"] == username:
                u["email"] = email
                u["phone"] = phone
                if new_password and new_password.strip():
                    u["password"] = bcrypt.hash(new_password)
                self.save_users(users)
                break

    # Replace the password for every user matching the given email.
    def update_password_by_email(self, email: str, new_password: str) -> bool:
        users = self.load_users()
        updated = False
        for u in users:
            if u.get("email") == email:
                u["password"] = bcrypt.hash(new_password)
                updated = True
        if updated:
            self.save_users(users)
        return updated

    # Find and return a user by username, or an empty dictionary if absent.
    def get_user_by_username(self, username: str) -> dict:
        for u in self.load_users():
            if u.get("username") == username:
                return u
        return {}