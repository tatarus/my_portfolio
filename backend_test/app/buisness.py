import random
from fastapi import APIRouter, Depends, HTTPException, status
from app.dependencies import get_current_user, require_role
from app.models import User

router = APIRouter(prefix="/projects", tags=["projects"])

owner_id=[1,2,3]
status=['active','plane','complete']
name=['Запуск сайта','Мобильное приложение', 'Админка']
secret_name=['Секретный сайт', 'Секретное приложение','Секретная разработка']
mock=[{'id':i+1, 'name':random.choice(name),'owner_id':random.choice(owner_id),'status':random.choice(status)} for i in range(10)]
secret_mock=[{'id':990+i,'name':random.choice(secret_name),'owner_id':999,'status':status} for i in range(10)]

@router.get("/")
def get_all_projects(current_user: User = Depends(get_current_user)):
    """ Доступно любому залогиненному"""
    return mock

@router.get("/admin-only")
def admin_projects(current_user: User = Depends(require_role("admin"))):
    """Только админ может видеть этот список."""
    return secret_mock   

@router.get("/{project_id}")
def get_project(project_id: int, current_user: User = Depends(get_current_user)):
    project = next((p for p in mock if p["id"] == project_id), None)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден")
    
    # Проверка прав: админу можно всё, пользователю – только свои проекты
    if current_user.role.name != "admin" and project["owner_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Нет доступа к этому проекту")
    
    return project