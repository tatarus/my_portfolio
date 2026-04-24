from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from app.database import Base
from sqlalchemy.orm import relationship

## для ТЗ п.2
class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)  # "admin", "user"
    description = Column(String, nullable=True)

    # связь с пользователями
    users = relationship("User", back_populates="role")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    patronymic = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    ## для ТЗ п.2
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False, default=2)  # по умолчанию role_id=2 (user)
    role = relationship("Role", back_populates="users")