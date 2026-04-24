from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from app import schemas, models, auth
from app.dependencies import get_db, get_current_user, require_role
from app.models import Role

router = APIRouter(prefix="/users", tags=["users"])

# Рег-я
@router.post("/register", response_model=schemas.UserOut)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    print('Запрос получен', user.dict())
    # проверка, что пароли совпадают
    if user.password != user.password2:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    # проверка, что email не занят
    existing = db.query(models.User).filter(models.User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    # создание пользователя
    hashed = auth.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        hashed_password=hashed,
        first_name=user.first_name,
        last_name=user.last_name,
        patronymic=user.patronymic,
        is_active=True
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # найти роль "user"
    user_role = db.query(Role).filter(Role.name == "user").first()
    db_user.role_id = user_role.id
    return db_user

# Логин -выдача токена
@router.post("/login")
# def login(email: str, password: str, db: Session = Depends(get_db)):
#     user = db.query(models.User).filter(models.User.email == email).first()
#     if not user or not auth.verify_password(password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     if not user.is_active:
#         raise HTTPException(status_code=401, detail="Account disabled")
#     token = auth.create_access_token(data={"sub": user.email})
#     return {"access_token": token, "token_type": "bearer"}
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = db.query(models.User).filter(models.User.email == form_data.username).first()
#     if not user or not auth.verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     if not user.is_active:
#         raise HTTPException(status_code=401, detail="Account disabled")
#     token = auth.create_access_token(data={"sub": user.email})
#     return {"access_token": token, "token_type": "bearer"}
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    email = form_data.username      
    password = form_data.password
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not auth.verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account disabled")
    token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}


# свой профиль
@router.get("/me", response_model=schemas.UserOut)
def read_me(current_user: models.User = Depends(get_current_user)):
    return current_user

# Обновить данные
@router.put("/me", response_model=schemas.UserOut)
def update_me(update_data: schemas.UserUpdate, db: Session = Depends(get_db),
              current_user: models.User = Depends(get_current_user)):
    for field, value in update_data.dict(exclude_unset=True).items():
        setattr(current_user, field, value)
    db.commit()
    db.refresh(current_user)
    return current_user

# Мягкое удаление
@router.delete("/me", status_code=204)
def delete_me(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    current_user.is_active = False
    db.commit()
    # logout - клиент сам удалит токен
    return

# Logout - отдельный эндпоинт (просто для удобства, ничего не делает на бэке)
@router.post("/logout")
def logout():
    return {"msg": "Logged out, please delete token on client"}

# Получить всех пользователей (только админ)
@router.get("/users", response_model=list[schemas.UserOut])
def get_all_users(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    users = db.query(models.User).filter(models.User.is_active == True).all()
    return users

## по ТЗ п.2
# Изменить роль пользователя (только админ)
@router.put("/users/{user_id}/role")
def change_user_role(
    user_id: int,
    role_name: str,  # "admin" или "user"
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    target_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not target_user:
        raise HTTPException(404, "Пользователь не найден")
    
    role = db.query(Role).filter(Role.name == role_name).first()
    if not role:
        raise HTTPException(400, "Нет такой роли")
    
    target_user.role_id = role.id
    db.commit()
    return {"msg": f"Роль изменена на {role_name}"}