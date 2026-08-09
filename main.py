from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.run import router as run_router
from routes.upload import router as upload_router
from routes.record import router as record_router
from routes.point_picker import router as point_picker_router
from routes.trial_review import router as trial_review_router
from routes.auth import router as auth_router
from db.init_db import init_db
from config import API_PREFIX

app = FastAPI(
    docs_url=f"{API_PREFIX}/docs",
    openapi_url=f"{API_PREFIX}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 修改為 * 以便調試，或是保留原本的
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix=API_PREFIX)
app.include_router(run_router, prefix=API_PREFIX)
app.include_router(upload_router, prefix=API_PREFIX)
app.include_router(record_router, prefix=API_PREFIX)
app.include_router(point_picker_router, prefix=API_PREFIX)
app.include_router(trial_review_router, prefix=API_PREFIX)

@app.on_event("startup")
async def on_startup():
    await init_db()
