// Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const selectFileBtn = document.getElementById('selectFileBtn');
const cameraBtn = document.getElementById('cameraBtn');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const processBtn = document.getElementById('processBtn');
const cancelBtn = document.getElementById('cancelBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');
const newSearchBtn = document.getElementById('newSearchBtn');

// Camera Modal Elements
const cameraModal = new bootstrap.Modal(document.getElementById('cameraModal'));
const cameraStream = document.getElementById('cameraStream');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');

// Result Elements
const plateImage = document.getElementById('plateImage');
const plateNumber = document.getElementById('plateNumber');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const vehicleDataContent = document.getElementById('vehicleDataContent');

// State
let selectedFile = null;
let currentStream = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

// Event Listeners Setup
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });

    // Select file button
    selectFileBtn.addEventListener('click', () => {
        imageInput.click();
    });

    // File input change
    imageInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Camera button
    cameraBtn.addEventListener('click', openCamera);

    // Capture button
    captureBtn.addEventListener('click', capturePhoto);

    // Process button
    processBtn.addEventListener('click', processImage);

    // Cancel button
    cancelBtn.addEventListener('click', resetUpload);

    // Retry button
    retryBtn.addEventListener('click', () => {
        hideAllSections();
        showSection(previewSection);
    });

    // New search button
    newSearchBtn.addEventListener('click', resetAll);

    // Modal close event
    document.getElementById('cameraModal').addEventListener('hidden.bs.modal', stopCamera);
}

// File Selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        if (validateFile(file)) {
            selectedFile = file;
            displayPreview(file);
        }
    }
}

// Drag and Drop Handlers
function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file && validateFile(file)) {
        selectedFile = file;
        imageInput.files = event.dataTransfer.files;
        displayPreview(file);
    }
}

// File Validation
function validateFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const maxSize = 16 * 1024 * 1024; // 16MB

    if (!validTypes.includes(file.type)) {
        showError('Por favor, selecciona una imagen válida (JPG o PNG)');
        return false;
    }

    if (file.size > maxSize) {
        showError('La imagen es demasiado grande. El tamaño máximo es 16MB');
        return false;
    }

    return true;
}

// Display Preview
function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        hideAllSections();
        showSection(previewSection);
    };
    reader.readAsDataURL(file);
}

// Camera Functions
async function openCamera() {
    try {
        // Check if mediaDevices is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError('Tu navegador no soporta acceso a la cámara. Usa Chrome, Firefox o Edge.');
            return;
        }

        // Try to get camera with rear camera preference (for mobile)
        const constraints = {
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        };

        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        cameraStream.srcObject = currentStream;
        
        // Wait for video to be ready
        cameraStream.onloadedmetadata = () => {
            cameraStream.play();
        };
        
        cameraModal.show();
    } catch (error) {
        console.error('Error accessing camera:', error);
        
        // More specific error messages
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            showError('Permiso de cámara denegado. Por favor, permite el acceso a la cámara en la configuración de tu navegador.');
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            showError('No se encontró ninguna cámara en tu dispositivo.');
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            showError('La cámara está siendo usada por otra aplicación. Cierra otras aplicaciones que puedan estar usando la cámara.');
        } else {
            showError('No se pudo acceder a la cámara: ' + error.message);
        }
    }
}

function capturePhoto() {
    if (!currentStream) {
        showError('No hay stream de cámara activo');
        return;
    }

    const context = cameraCanvas.getContext('2d');
    
    // Set canvas size to match video
    cameraCanvas.width = cameraStream.videoWidth;
    cameraCanvas.height = cameraStream.videoHeight;
    
    // Draw current video frame to canvas
    context.drawImage(cameraStream, 0, 0, cameraCanvas.width, cameraCanvas.height);
    
    // Convert canvas to blob
    cameraCanvas.toBlob((blob) => {
        if (!blob) {
            showError('Error al capturar la foto');
            return;
        }

        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        selectedFile = file;
        
        // Create a FileList-like object
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        imageInput.files = dataTransfer.files;
        
        displayPreview(file);
        cameraModal.hide();
    }, 'image/jpeg', 0.95);
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        cameraStream.srcObject = null;
        currentStream = null;
    }
}

// Process Image
async function processImage() {
    if (!selectedFile) {
        showError('No hay imagen seleccionada');
        return;
    }

    hideAllSections();
    showSection(loadingSection);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch(`${API_BASE_URL}/process-plate`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            handleProcessResponse(data);
        } else {
            showError(data.message || 'Error al procesar la imagen');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Error de conexión con el servidor. Verifica que el backend esté ejecutándose.');
    }
}

// Handle Process Response
function handleProcessResponse(data) {
    hideAllSections();

    if (data.status === 'success') {
        displayResults(data);
    } else if (data.status === 'no_plate_detected') {
        showError(data.message);
    } else if (data.status === 'multiple_plates_detected') {
        showError(data.message);
    } else if (data.status === 'no_text_detected') {
        if (data.plate_image) {
            plateImage.src = `data:image/jpeg;base64,${data.plate_image}`;
        }
        showError(data.message);
    } else if (data.status === 'low_confidence' || data.status === 'invalid_format') {
        displayResults(data, true); // Show with warning
    } else {
        showError(data.message || 'Error desconocido al procesar la imagen');
    }
}

// Display Results
function displayResults(data, isWarning = false) {
    // Display plate image
    if (data.plate_image) {
        plateImage.src = `data:image/jpeg;base64,${data.plate_image}`;
    }

    // Display plate number
    plateNumber.textContent = data.plate_number || '---';

    // Display confidence
    const confidence = Math.round((data.confidence || 0) * 100);
    confidenceBar.style.width = `${confidence}%`;
    confidenceBar.setAttribute('aria-valuenow', confidence);
    confidenceText.textContent = `${confidence}%`;

    // Set confidence bar color
    if (confidence >= 80) {
        confidenceBar.className = 'progress-bar bg-success';
    } else if (confidence >= 60) {
        confidenceBar.className = 'progress-bar bg-warning';
    } else {
        confidenceBar.className = 'progress-bar bg-danger';
    }

    // Display vehicle data
    displayVehicleData(data);

    showSection(resultsSection);

    // Show warning if needed
    if (isWarning) {
        showWarningBanner(data.message);
    }
}

// Display Vehicle Data
function displayVehicleData(data) {
    vehicleDataContent.innerHTML = '';

    if (data.vehicle_data) {
        const vehicleData = data.vehicle_data;
        
        const gridHTML = `
            <div class="vehicle-info-grid">
                <div class="vehicle-info-item">
                    <div class="vehicle-info-label">
                        <i class="bi bi-calendar-check me-1"></i>
                        Año
                    </div>
                    <div class="vehicle-info-value">${vehicleData.year || 'N/A'}</div>
                </div>
                <div class="vehicle-info-item">
                    <div class="vehicle-info-label">
                        <i class="bi bi-car-front me-1"></i>
                        Tipo
                    </div>
                    <div class="vehicle-info-value">${vehicleData.type || 'N/A'}</div>
                </div>
                <div class="vehicle-info-item">
                    <div class="vehicle-info-label">
                        <i class="bi bi-tag me-1"></i>
                        Subtipo
                    </div>
                    <div class="vehicle-info-value">${vehicleData.subtype || 'N/A'}</div>
                </div>
                <div class="vehicle-info-item">
                    <div class="vehicle-info-label">
                        <i class="bi bi-building me-1"></i>
                        Marca
                    </div>
                    <div class="vehicle-info-value">${vehicleData.make || 'N/A'}</div>
                </div>
                <div class="vehicle-info-item" style="grid-column: span 2;">
                    <div class="vehicle-info-label">
                        <i class="bi bi-card-text me-1"></i>
                        Modelo
                    </div>
                    <div class="vehicle-info-value">${vehicleData.model || 'N/A'}</div>
                </div>
            </div>
        `;

        vehicleDataContent.innerHTML = gridHTML;

        // Add vehicle image if available
        if (vehicleData.image_url) {
            const imageHTML = `
                <div class="vehicle-image-container">
                    <h5 class="text-muted mb-3">Imagen del Vehículo</h5>
                    <img src="${vehicleData.image_url}" alt="Vehículo" 
                         onerror="this.parentElement.style.display='none'">
                </div>
            `;
            vehicleDataContent.innerHTML += imageHTML;
        }
    } else {
        // No vehicle data found
        const noDataHTML = `
            <div class="no-data-message">
                <i class="bi bi-info-circle-fill mb-3"></i>
                <h5>No se encontraron datos del vehículo</h5>
                <p class="mb-0">La placa fue detectada correctamente, pero no se encontró información 
                del vehículo en la base de datos.</p>
            </div>
        `;
        vehicleDataContent.innerHTML = noDataHTML;
    }
}

// Show Warning Banner
function showWarningBanner(message) {
    const banner = document.createElement('div');
    banner.className = 'alert alert-warning alert-dismissible fade show';
    banner.innerHTML = `
        <i class="bi bi-exclamation-triangle-fill me-2"></i>
        <strong>Advertencia:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    resultsSection.insertBefore(banner, resultsSection.firstChild);
}

// Section Management
function hideAllSections() {
    previewSection.classList.add('d-none');
    loadingSection.classList.add('d-none');
    resultsSection.classList.add('d-none');
    errorSection.classList.add('d-none');
}

function showSection(section) {
    section.classList.remove('d-none');
}

// Error Display
function showError(message) {
    hideAllSections();
    errorMessage.textContent = message;
    showSection(errorSection);
}

// Reset Functions
function resetUpload() {
    selectedFile = null;
    imageInput.value = '';
    imagePreview.src = '';
    hideAllSections();
}

function resetAll() {
    resetUpload();
    plateImage.src = '';
    plateNumber.textContent = '---';
    confidenceBar.style.width = '0%';
    confidenceText.textContent = '0%';
    vehicleDataContent.innerHTML = '';
}
