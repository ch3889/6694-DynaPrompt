# Alternative: Simplified Terraform (No Service Account Issues)

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Variables
variable "project_id" {
  description = "Your GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

# Deep Learning VM Image
data "google_compute_image" "deep_learning" {
  family  = "pytorch-latest-gpu"
  project = "deeplearning-platform-release"
}

# Simplified compute instance
resource "google_compute_instance" "dynaprompt_gpu" {
  name         = "dynaprompt-dl"
  machine_type = "n1-standard-4"
  zone         = var.zone

  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = data.google_compute_image.deep_learning.self_link
      size  = 100
      type  = "pd-standard"
    }
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
    preemptible        = false
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  tags = ["http-server", "https-server"]
}

# Outputs
output "instance_name" {
  value = google_compute_instance.dynaprompt_gpu.name
}

output "external_ip" {
  value = google_compute_instance.dynaprompt_gpu.network_interface[0].access_config[0].nat_ip
}

output "ssh_command" {
  value = "gcloud compute ssh ${google_compute_instance.dynaprompt_gpu.name} --zone=${var.zone}"
}
