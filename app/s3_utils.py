"""
Minimal S3 Utilities for RAG Applications
Functionalities Included:
- Upload PDF to S3
- Download PDF from S3
- List documents in S3
- Process uploaded files (extract text + upload)
"""

import boto3
from botocore.exceptions import ClientError
import os
from typing import List, Dict, Any
from io import BytesIO
from app.pdf_utils import clean_text

# -------------------------
# S3 CONFIG
# -------------------------
S3_CONFIG = {
    "access_key": os.getenv("S3_ACCESS_KEY", ""),
    "secret_key": os.getenv("S3_SECRET_KEY", ""),
    "bucket": os.getenv("S3_BUCKET_NAME", "medibot-bucket"),
    "region": os.getenv("S3_REGION", "us-east-1")
}

# -------------------------
# CLIENT
# -------------------------
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=S3_CONFIG["access_key"],
        aws_secret_access_key=S3_CONFIG["secret_key"],
        region_name=S3_CONFIG["region"]
    )

# -------------------------
# UPLOAD TO S3
# -------------------------
def upload_to_s3(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    s3 = get_s3_client()
    key = f"documents/{filename}"

    try:
        s3.put_object(
            Bucket=S3_CONFIG["bucket"],
            Key=key,
            Body=file_bytes,
            ContentType="application/pdf"
        )
        return {"success": True, "s3_key": key}
    except ClientError as e:
        return {"success": False, "error": str(e)}

# -------------------------
# DOWNLOAD FROM S3
# -------------------------
def download_from_s3(s3_key: str) -> Dict[str, Any]:
    s3 = get_s3_client()

    try:
        response = s3.get_object(
            Bucket=S3_CONFIG["bucket"],
            Key=s3_key
        )
        return {
            "success": True,
            "content": response["Body"].read(),
            "filename": s3_key.split("/")[-1]
        }
    except ClientError as e:
        return {"success": False, "error": str(e)}

# -------------------------
# LIST ALL DOCUMENTS
# -------------------------
def list_s3_documents() -> List[Dict[str, Any]]:
    s3 = get_s3_client()

    try:
        response = s3.list_objects_v2(
            Bucket=S3_CONFIG["bucket"],
            Prefix="documents/"
        )
        documents = []

        for obj in response.get("Contents", []):
            documents.append({
                "key": obj["Key"],
                "filename": obj["Key"].split("/")[-1],
                "size": obj["Size"],
                "last_modified": obj["LastModified"]
            })

        return documents

    except Exception:
        return []

# -------------------------
# PROCESS UPLOADED FILES (Extract Text + Upload)
# -------------------------
def process_uploaded_files(uploaded_files: List, extract_text_func):
    results = {"uploaded": [], "failed": [], "texts": []}

    for file in uploaded_files:
        try:
            file_bytes = file.read()
            text = clean_text(extract_text_func(BytesIO(file_bytes)))

            # Upload to S3
            upload_result = upload_to_s3(file_bytes, file.name)

            if upload_result["success"]:
                results["uploaded"].append({
                    "filename": file.name,
                    "s3_key": upload_result["s3_key"]
                })
            else:
                results["failed"].append({
                    "filename": file.name,
                    "error": upload_result["error"]
                })

            results["texts"].append(text)

        except Exception as e:
            results["failed"].append({"filename": file.name, "error": str(e)})

    return results
