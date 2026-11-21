###############################################################################
# RAG Course Assistant — Infrastructure as Code
#
# Terraform owns: 2 Cloud Run services + 2 Cloud Build triggers + IAM bindings
# Terraform reads: GCS bucket + Secret Manager secrets (data sources — never
#                  created/modified/destroyed)
# Cloud Build owns: the container image (push → build → new revision)
#
# Services use the v1 (knative-style) resource — matches the schema the
# Console creates and avoids the v2 multi-region bug in provider 6.49/6.50.
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  type    = string
  default = "rag-course-assistant-514120"
}

variable "region" {
  type    = string
  default = "asia-south1"
}

locals {
  runtime_sa = "934403447461-compute@developer.gserviceaccount.com"
  ar_repo    = "asia-south1-docker.pkg.dev/${var.project_id}/cloud-run-source-deploy"
}

# --- existing stateful resources: READ-ONLY (Terraform never touches) -------

data "google_storage_bucket" "index" {
  name = "rag-index-514120"
}

data "google_secret_manager_secret" "nvidia_key" {
  secret_id = "NVIDIA_API_KEY"
  version = "1"

}

data "google_secret_manager_secret" "google_key" {
  secret_id = "GOOGLE_API_KEY"
  version = "1"
}

# --- rag-api (v1 knative schema — same shape the Console produces) ------------

resource "google_cloud_run_service" "rag_api" {
  name     = "rag-api"
  location = var.region

  metadata {
    annotations = {
      "run.googleapis.com/ingress"  = "all"
      "run.googleapis.com/maxScale" = "5"
    }
  }

  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale"     = "0"
        "run.googleapis.com/startup-cpu-boost" = "true"
      }
    }
    spec {
      container_concurrency = 10
      timeout_seconds       = 900
      service_account_name  = local.runtime_sa
      containers {
        image = "asia-south1-docker.pkg.dev/rag-course-assistant-514120/cloud-run-source-deploy/rag-course-assistant/rag-api:cf114416ad0231e924925872c7781bf0c285b8cb"
        ports {
          container_port = 8080
        }
        resources {
          limits = {
            cpu    = "1"
            memory = "2Gi"
          }
        }
        env {
          name  = "FAISS_INDEX_PATH"
          value = "/tmp/faiss.idx"
        }
        env {
          name  = "CHUNKS_META_PATH"
          value = "/tmp/chunks.json"
        }
        env {
          name  = "GCS_INDEX_BUCKET"
          value = "rag-index-514120"
        }
        # Retrieval params — calibrated via eval sweep (REPORT §19): 0.70 is
        # the only threshold with refusal accuracy 1.0 AND hit@3 1.0.
        # Mirrors .env — keep in sync (this was missing pre-fix: the deployed
        # service silently ran the uncalibrated 0.35, breaking refusals).
        env {
          name  = "TOP_K"
          value = "3"
        }
        env {
          name  = "RELEVANCE_THRESHOLD"
          value = "0.70"
        }
        env {
          name  = "TEMPERATURE"
          value = "0.2"
        }
        env {
          name  = "LLM_PROVIDER"
          value = "nvidia"
        }
        env {
          name  = "NVIDIA_MODEL"
          value = "z-ai/glm-5.3-flash"
        }
        env {
          name  = "MAX_TOKENS"
          value = "768"
        }
        env {
          name = "NVIDIA_API_KEY"
          value_from {
            secret_key_ref {
              name = "NVIDIA_API_KEY"
              key  = "latest"
            }
          }
        }
        env {
          name = "GOOGLE_API_KEY"
          value_from {
            secret_key_ref {
              name = "GOOGLE_API_KEY"
              key  = "latest"
            }
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  # Cloud Build owns the image + build labels; TF owns everything else.
  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image,
      template[0].metadata[0].labels,
      metadata[0].annotations, # gcloud adds client-name/client-version on deploy
      metadata[0].labels,
    ]
  }
}

# --- rag-ui ---------------------------------------------------------------------

resource "google_cloud_run_service" "rag_ui" {
  name     = "rag-ui"
  location = var.region

  metadata {
    annotations = {
      "run.googleapis.com/ingress"  = "all"
      "run.googleapis.com/maxScale" = "5"
    }
  }

  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale"     = "0"
        "run.googleapis.com/startup-cpu-boost" = "true"
      }
    }
    spec {
      container_concurrency = 80
      timeout_seconds       = 3600
      service_account_name  = local.runtime_sa
      containers {
        image = "asia-south1-docker.pkg.dev/rag-course-assistant-514120/cloud-run-source-deploy/rag-course-assistant/rag-ui:cf114416ad0231e924925872c7781bf0c285b8cb"
        ports {
          container_port = 8080
        }
        resources {
          limits = {
            cpu    = "1"
            memory = "1Gi"
          }
        }
        env {
          name  = "API_URL"
          value = "https://rag-api-934403447461.asia-south1.run.app"
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image,
      template[0].metadata[0].labels,
      metadata[0].annotations, # gcloud adds client-name/client-version on deploy
      metadata[0].labels,
    ]
  }
}

# --- public access (unauthenticated demo requirement) ---------------------------

resource "google_cloud_run_service_iam_member" "rag_api_public" {
  service  = google_cloud_run_service.rag_api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "rag_ui_public" {
  service  = google_cloud_run_service.rag_ui.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --- Cloud Build triggers: push to main → build → deploy ------------------------
# 1st-gen GitHub App triggers live in the global region.

resource "google_cloudbuild_trigger" "rag_api_deploy" {
  name            = "rag-api-deploy"
  location        = "global"
  description     = "Build and deploy to Cloud Run service rag-api on push to main"
  service_account = "projects/rag-course-assistant-514120/serviceAccounts/934403447461-compute@developer.gserviceaccount.com"

  github {
    owner = "Shubhampm18"
    name  = "rag-course-assistant"
    push {
      branch = "^main$"
    }
  }

  filename = "/infra/cloudbuild-deploy-api.yaml"

  substitutions = {
    _AR_HOSTNAME   = "asia-south1-docker.pkg.dev"
    _AR_PROJECT_ID = var.project_id
    _AR_REPOSITORY = "cloud-run-source-deploy"
    _DEPLOY_REGION = var.region
    _PLATFORM      = "managed"
    _SERVICE_NAME  = "rag-api"
  }
}

resource "google_cloudbuild_trigger" "rag_ui_deploy" {
  name            = "rag-ui-deploy"
  location        = "global"
  description     = "Build and deploy to Cloud Run service rag-ui on push to main"
  service_account = "projects/rag-course-assistant-514120/serviceAccounts/934403447461-compute@developer.gserviceaccount.com"

  github {
    owner = "Shubhampm18"
    name  = "rag-course-assistant"
    push {
      branch = "^main$"
    }
  }

  filename = "/infra/cloudbuild-deploy-ui.yaml"

  substitutions = {
    _AR_HOSTNAME   = "asia-south1-docker.pkg.dev"
    _AR_PROJECT_ID = var.project_id
    _AR_REPOSITORY = "cloud-run-source-deploy"
    _DEPLOY_REGION = var.region
    _PLATFORM      = "managed"
    _SERVICE_NAME  = "rag-ui"
  }
}

# --- IAM bindings (re-assert the grants the services need) ----------------------

resource "google_secret_manager_secret_iam_member" "nvidia_accessor" {
  secret_id = data.google_secret_manager_secret.nvidia_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${local.runtime_sa}"
}

resource "google_secret_manager_secret_iam_member" "google_accessor" {
  secret_id = data.google_secret_manager_secret.google_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${local.runtime_sa}"
}

resource "google_storage_bucket_iam_member" "index_creator" {
  bucket = data.google_storage_bucket.index.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${local.runtime_sa}"
}
