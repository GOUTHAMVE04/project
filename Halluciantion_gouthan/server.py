import os
import re
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PIL import Image

from config import Config
from pipeline.hallucination_verifier import DescriptionHallucinationVerifier
from verify_description import check_hallucination_and_pope

# Initialize FastAPI App
app = FastAPI(
    title="ICD Hallucination & POPE Verifier Web Server",
    description="Instruction Contrastive Decoding (Wang et al., ACL 2024) Multimodal Verification Service",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(PROJECT_DIR, "web")
RESULTS_DIR = Config.RESULTS_DIR
UPLOADS_DIR = os.path.join(RESULTS_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mount Static Directories
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/samples", StaticFiles(directory=PROJECT_DIR), name="samples")

# Global Preloaded Verifier Engine (Singleton for sub-second responses)
print("\n[Server Startup] Preloading Vision-Language ICD Verifier Engine into memory...")
GLOBAL_VERIFIER = DescriptionHallucinationVerifier()
print("[Server Startup] ICD Verifier Engine ready for instant queries!\n")


@app.get("/")
async def serve_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Web interface index.html not found.")


@app.get("/api/samples")
async def list_sample_images():
    """Returns sample images and preset captions available in the workspace."""
    samples = [
        {
            "name": "Lone Tree",
            "file": "tree.jpg",
            "description": "A person and tree is there",
            "thumb": "/samples/tree.jpg"
        },
        {
            "name": "Cat on Fence",
            "file": "images (1).jpg",
            "description": "human is taking photo and cat is there",
            "thumb": "/samples/images%20(1).jpg"
        },
        {
            "name": "Hot Air Balloon",
            "file": "download.jpg",
            "description": "a balloon flying over the canyon",
            "thumb": "/samples/download.jpg"
        },
        {
            "name": "Mountain Sky",
            "file": "image.png",
            "description": "A bird is flying between mountains",
            "thumb": "/samples/image.png"
        }
    ]
    return {"samples": samples}


@app.post("/api/verify")
async def verify_image_endpoint(
    description: str = Form(...),
    image_file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None)
):
    """
    Main verification API:
    Accepts uploaded image file OR sample image name, along with user description.
    Runs ICD hallucination verification and image-specific POPE evaluation.
    """
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="Description text is required.")

    target_image_path = None

    # Handle Uploaded File
    if image_file and image_file.filename:
        safe_filename = re.sub(r"[^\w\-.]", "_", image_file.filename)
        dest_path = os.path.join(UPLOADS_DIR, safe_filename)
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        target_image_path = dest_path

    # Handle Sample Image Name
    elif sample_name:
        candidate = os.path.join(PROJECT_DIR, sample_name)
        if os.path.exists(candidate):
            target_image_path = candidate
        else:
            raise HTTPException(status_code=404, detail=f"Sample image '{sample_name}' not found.")
    else:
        # Fallback to default image
        fallback = os.path.join(PROJECT_DIR, "tree.jpg")
        if os.path.exists(fallback):
            target_image_path = fallback
        else:
            raise HTTPException(status_code=400, detail="Please upload an image or select a sample image.")

    try:
        # Run verification using the preloaded resident verifier
        result = check_hallucination_and_pope(
            image_path=target_image_path,
            description=description.strip(),
            model_mode="proposed",
            run_pope=True,
            save_visual=True,
            verifier_instance=GLOBAL_VERIFIER
        )

        base_name = os.path.splitext(os.path.basename(target_image_path))[0]
        safe_name = re.sub(r"[^\w\-]", "_", base_name)

        comp_filename = f"{safe_name}_perturbed_comparison.png"
        report_filename = f"{safe_name}_hallucination_report.json"

        return {
            "status": "success",
            "image_name": os.path.basename(target_image_path),
            "description": description.strip(),
            "verification": result["verification"],
            "pope_results": result["pope_results"],
            "comparison_image_url": f"/results/{comp_filename}",
            "report_json_url": f"/results/{report_filename}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def start_server(port: int = 8000, host: str = "127.0.0.1"):
    print("=" * 72)
    print("      ICD HALLUCINATION & POPE VERIFICATION WEB SERVER")
    print("=" * 72)
    print(f"Server URL:     http://{host}:{port}")
    print(f"API Docs:       http://{host}:{port}/docs")
    print(f"Results Mount:  http://{host}:{port}/results")
    print("=" * 72)
    uvicorn.run("server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    import sys
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port=port)
