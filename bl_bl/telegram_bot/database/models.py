from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    created: Mapped[DateTime] = mapped_column(DateTime, default=func.now())
    updated: Mapped[DateTime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())


class User(Base):
    __tablename__ = "table_user"

    tg_id: Mapped[int] = mapped_column(primary_key=True)

    requests_used: Mapped[int] = mapped_column(default=0, nullable=False)
    duration_used: Mapped[int] = mapped_column(default=0, nullable=False)

    requests_max: Mapped[int] = mapped_column(default=10000, nullable=False)
    duration_max: Mapped[int] = mapped_column(default=60000, nullable=False)

    last_request: Mapped[DateTime] = mapped_column(DateTime, default=func.now(), nullable=False)
