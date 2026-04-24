# init_db.py
from app.database import SessionLocal
from app.models import User, Role
from app.auth import get_password_hash

def init_roles_and_admin():
    db = SessionLocal()
    
    # 1. Создаём роли (если их нет)
    roles = ["admin", "user"]
    for role_name in roles:
        role = db.query(Role).filter(Role.name == role_name).first()
        if not role:
            role = Role(name=role_name, description=f"{role_name} role")
            db.add(role)
    db.commit()
    
    # 2. Получаем id ролей
    admin_role = db.query(Role).filter(Role.name == "admin").first()
    user_role = db.query(Role).filter(Role.name == "user").first()
    
    # 3. Создаём администратора (если не существует)
    admin_email = "admin@example.com"
    admin = db.query(User).filter(User.email == admin_email).first()
    if not admin:
        admin = User(
            email=admin_email,
            hashed_password=get_password_hash("admin123"),
            first_name="Admin",
            last_name="Super",
            patronymic=None,
            is_active=True,
            role_id=admin_role.id
        )
        db.add(admin)
    
    # 4. Убедимся, что у всех существующих пользователей есть роль (user)
    users = db.query(User).filter(User.role_id == None).all()
    for u in users:
        u.role_id = user_role.id
    
    db.commit()
    db.close()
    print("База данных инициализирована: роли и админ созданы.")

if __name__ == "__main__":
    init_roles_and_admin()