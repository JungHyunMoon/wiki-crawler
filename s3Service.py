import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def s3_connection():
    try:
        # s3 클라이언트 생성
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3


def s3_upload(fileName, file):
    s3 = s3_connection()
    try:
        # 파일 업로드 시 Content-Type과 Content-Disposition 설정
        s3.put_object(
            Body=file,
            Bucket="ragresource",
            Key=fileName,
            ContentType="application/pdf; charset=utf-8",  # UTF-8 인코딩 설정 추가
            ContentDisposition="inline",  # 파일을 브라우저에서 미리보기로 표시
        )
    except Exception as e:
        print(e)