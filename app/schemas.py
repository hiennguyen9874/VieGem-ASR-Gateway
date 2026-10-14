"""Wire schemas shared by REST, SSE and realtime adapters (docs/api-contract.vi.md)."""

from pydantic import BaseModel


class ErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class Status(BaseModel):
    status: str
