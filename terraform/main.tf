# Terraform configuration for DynaPrompt on GCP with GPU
# Optimized for $300 free trial credits

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

variable "machine_type" {
  description = "Machine type for the VM"
  type        = string
  default     = "n1-standard-4"
}

variable "gpu_type" {
  description = "GPU type"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 1
}

variable "disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 100
}

# Deep Learning VM Image (PyTorch 2.0 with CUDA 11.8)
data "google_compute_image" "deep_learning" {
  family  = "pytorch-latest-gpu"
  project = "deeplearning-platform-release"
}

# Compute Instance with GPU
resource "google_compute_instance" "dynaprompt_gpu" {
  name         = "dynaprompt-dl"
  machine_type = var.machine_type
  zone         = var.zone

  # Allow instance to be stopped for GPU attachment
  allow_stopping_for_update = true

  # Boot disk
  boot_disk {
    initialize_params {
      image = data.google_compute_image.deep_learning.self_link
      size  = var.disk_size_gb
      type  = "pd-standard"  # Use pd-ssd for faster disk
    }
  }

  # GPU configuration
  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  # Required for GPU
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
    preemptible        = false  # Set to true for 70% cost savings
  }

  # Network interface
  network_interface {
    network = "default"
    access_config {
      # Ephemeral public IP
    }
  }

  # Metadata
  metadata = {
    install-nvidia-driver = "True"
    proxy-mode           = "project_editors"
  }

  # Startup script
  metadata_startup_script = <<-EOF
    #!/bin/bash
    
    # Install additional dependencies
    pip install transformers==4.35.0 diffusers==0.24.0 accelerate==0.25.0
    pip install omegaconf==2.3.0 einops==0.7.0 kornia==0.7.0
    pip install pytorch-lightning==2.1.0 torchmetrics==1.2.0
    pip install tqdm matplotlib
    
    # Clone repository
    cd /home/$(ls /home | head -1)
    if [ ! -d "6694-DynaPrompt" ]; then
      sudo -u $(ls /home | head -1) git clone https://github.com/ch3889/6694-DynaPrompt.git
      cd 6694-DynaPrompt
      sudo -u $(ls /home | head -1) git checkout zk2295
    fi
    
    # Log completion
    echo "Setup complete at $(date)" > /tmp/setup_complete.txt
  EOF

  # Firewall tags
  tags = ["http-server", "https-server"]

  # Service account - use default Compute Engine service account
  service_account {
    # Uses default: PROJECT_NUMBER-compute@developer.gserviceaccount.com
    email  = data.google_compute_default_service_account.default.email
    scopes = ["cloud-platform"]
  }

  # Labels for cost tracking
  labels = {
    project     = "dynaprompt"
    environment = "development"
    gpu         = "t4"
  }
}

# Get default Compute Engine service account
data "google_compute_default_service_account" "default" {
}

# Firewall rule for HTTP
resource "google_compute_firewall" "allow_http" {
  name    = "allow-http-dynaprompt"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server"]
}

# Firewall rule for HTTPS
resource "google_compute_firewall" "allow_https" {
  name    = "allow-https-dynaprompt"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["https-server"]
}

# Optional: Firewall rule for Jupyter (port 8888)
resource "google_compute_firewall" "allow_jupyter" {
  name    = "allow-jupyter-dynaprompt"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8888"]
  }

  source_ranges = ["0.0.0.0/0"]  # Restrict to your IP for security
  target_tags   = ["jupyter-server"]
}

# Outputs
output "instance_name" {
  description = "Name of the compute instance"
  value       = google_compute_instance.dynaprompt_gpu.name
}

output "instance_zone" {
  description = "Zone of the compute instance"
  value       = google_compute_instance.dynaprompt_gpu.zone
}

output "external_ip" {
  description = "External IP address"
  value       = google_compute_instance.dynaprompt_gpu.network_interface[0].access_config[0].nat_ip
}

output "ssh_command" {
  description = "Command to SSH into the instance"
  value       = "gcloud compute ssh ${google_compute_instance.dynaprompt_gpu.name} --zone=${var.zone}"
}

output "estimated_cost_per_hour" {
  description = "Estimated cost per hour (USD)"
  value       = "~$0.54/hour (n1-standard-4 + T4 GPU)"
}

output "setup_status" {
  description = "Check setup completion"
  value       = "gcloud compute ssh ${google_compute_instance.dynaprompt_gpu.name} --zone=${var.zone} --command='cat /tmp/setup_complete.txt'"
}
