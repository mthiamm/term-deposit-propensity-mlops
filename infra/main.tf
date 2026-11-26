terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

variable "localstack" {
  description = "Talk to LocalStack on localhost instead of real AWS"
  type        = bool
  default     = true
}

variable "bucket_name" {
  type    = string
  default = "term-deposit-models"
}

provider "aws" {
  region = "us-east-1"

  # These only apply when talking to LocalStack
  access_key                  = var.localstack ? "test" : null
  secret_key                  = var.localstack ? "test" : null
  skip_credentials_validation = var.localstack
  skip_metadata_api_check     = var.localstack
  skip_requesting_account_id  = var.localstack
  s3_use_path_style           = var.localstack

  endpoints {
    s3 = var.localstack ? "http://localhost:4566" : null
  }
}

resource "aws_s3_bucket" "models" {
  bucket = var.bucket_name
}

# Keep every previous model version when an object is overwritten
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Models are private, so block every form of public access
resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "bucket_name" {
  value = aws_s3_bucket.models.bucket
}