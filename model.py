# # models.py

# from sqlalchemy import Column, Integer, String
# from db import Base


# # User model representing the user table
# class reg(Base):
#     __tablename__ = "signin"

#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, unique=True, index=True)
#     password = Column(String)


# class enrty(Base):
#     __tablename__ = "sinup"

#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, unique=True, index=True)
#     password = Column(String)
#     full_name = Column(String, index=True)
# models.py

# from sqlalchemy import Column, Integer, String
# from db import Base


# # User model representing the user table
# class reg(Base):
#     __tablename__ = "signin"

#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, unique=True, index=True)
#     password = Column(String)


# class enrty(Base):
#     __tablename__ = "sinup"

#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, unique=True, index=True)
#     password = Column(String)
#     full_name = Column(String, index=True)


from sqlalchemy import Column, DateTime, Integer, String
from db import Base


# User model representing the user table
# class reg(Base):
#     __tablename__ = "signin"

#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, unique=True, index=True)
#     password = Column(String)


class enrty(Base):
    __tablename__ = "sinup"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    full_name = Column(String, index=True)
    credits = Column(Integer, default=100)

from sqlalchemy import func

class JobSubmission(Base):
    __tablename__ = "job_submissions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    job_title = Column(String, index=True)
    description = Column(String)
    deadline = Column(DateTime)
    priority = Column(String)
    submission_time = Column(DateTime, server_default=func.now())
