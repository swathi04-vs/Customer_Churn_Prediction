/**
 * Main JavaScript File for Customer Churn Prediction Application
 * Handles UI interactions, form submissions, and AJAX requests
 */

// ==================== Utility Functions ====================

/**
 * Show a flash message on the screen
 */
function showFlashMessage(message, type = 'success') {
    const alertType = type === 'error' ? 'danger' : type;
    const alertHTML = `
        <div class="alert alert-${alertType} alert-dismissible fade show m-3" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    const mainElement = document.querySelector('main');
    if (mainElement) {
        mainElement.insertAdjacentHTML('afterbegin', alertHTML);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = mainElement.querySelector('.alert');
            if (alert) {
                alert.remove();
            }
        }, 5000);
    }
}

/**
 * Validate email format
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Format a number as currency
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

/**
 * Debounce function for performance optimization
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ==================== API Functions ====================

/**
 * Fetch API endpoint with error handling
 */
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Call Error:', error);
        throw error;
    }
}

/**
 * Get model information
 */
async function getModelInfo() {
    try {
        return await apiCall('/api/model-info');
    } catch (error) {
        console.error('Error fetching model info:', error);
        throw error;
    }
}

/**
 * Make prediction
 */
async function makePrediction(data) {
    try {
        return await apiCall('/api/predict', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    } catch (error) {
        console.error('Error making prediction:', error);
        throw error;
    }
}

/**
 * Check API health
 */
async function checkHealth() {
    try {
        return await apiCall('/api/health');
    } catch (error) {
        console.error('Error checking health:', error);
        return { status: 'unhealthy' };
    }
}

// ==================== DOM Manipulation ====================

/**
 * Initialize tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize popovers
 */
function initializePopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

/**
 * Add active class to current nav item
 */
function setActiveNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar .nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// ==================== Form Handling ====================

/**
 * Clear form validation
 */
function clearFormValidation(form) {
    const inputs = form.querySelectorAll('.is-invalid');
    inputs.forEach(input => {
        input.classList.remove('is-invalid');
    });
}

/**
 * Show form validation error
 */
function showFormError(input, message) {
    input.classList.add('is-invalid');
    const feedback = input.parentElement.querySelector('.invalid-feedback');
    if (feedback) {
        feedback.textContent = message;
        feedback.style.display = 'block';
    }
}

/**
 * Validate form
 */
function validateForm(form) {
    clearFormValidation(form);
    let isValid = true;
    const inputs = form.querySelectorAll('input[required], select[required]');
    
    inputs.forEach(input => {
        if (!input.value) {
            showFormError(input, 'This field is required');
            isValid = false;
        }
    });
    
    return isValid;
}

// ==================== Event Listeners ====================

/**
 * Initialize event listeners
 */
function initializeEventListeners() {
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Ctrl+Enter to submit form
        if (event.ctrlKey && event.key === 'Enter') {
            const form = document.querySelector('form');
            if (form) {
                form.dispatchEvent(new Event('submit'));
            }
        }
    });
}

// ==================== Page Initialization ====================

/**
 * Initialize page
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page initialized');
    
    // Initialize Bootstrap components
    initializeTooltips();
    initializePopovers();
    
    // Set active nav item
    setActiveNavItem();
    
    // Initialize event listeners
    initializeEventListeners();
    
    // Check API health
    checkHealth().then(result => {
        console.log('API Health:', result.status);
        if (result.status === 'unhealthy') {
            console.warn('API is not fully initialized');
        }
    });
});

// ==================== Page Unload ====================

/**
 * Cleanup on page unload
 */
window.addEventListener('beforeunload', function() {
    // You can add cleanup code here
});

// ==================== Error Handling ====================

/**
 * Global error handler
 */
window.addEventListener('error', function(event) {
    console.error('Global Error:', event.error);
    showFlashMessage('An unexpected error occurred', 'error');
});

/**
 * Handle unhandled promise rejections
 */
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled Rejection:', event.reason);
    showFlashMessage('An unexpected error occurred', 'error');
});

// ==================== Utility Classes ====================

class Toast {
    /**
     * Show a toast notification
     */
    static show(message, type = 'info', duration = 3000) {
        const toastHTML = `
            <div class="toast align-items-center text-white bg-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info'} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        const toastContainer = document.querySelector('.toast-container') || 
            (() => {
                const container = document.createElement('div');
                container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
                document.body.appendChild(container);
                return container;
            })();
        
        const toastElement = document.createElement('div');
        toastElement.innerHTML = toastHTML;
        toastContainer.appendChild(toastElement.firstElementChild);
        
        const toast = new bootstrap.Toast(toastElement.firstElementChild);
        toast.show();
        
        setTimeout(() => {
            toastElement.firstElementChild.remove();
        }, duration);
    }
}

class Loader {
    /**
     * Show loading spinner
     */
    static show(message = 'Loading...') {
        const loaderHTML = `
            <div class="spinner-overlay">
                <div class="spinner-content">
                    <div class="spinner-border text-primary mb-3" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="text-muted">${message}</p>
                </div>
            </div>
        `;
        
        if (!document.querySelector('.spinner-overlay')) {
            document.body.insertAdjacentHTML('beforeend', loaderHTML);
        }
    }
    
    /**
     * Hide loading spinner
     */
    static hide() {
        const spinner = document.querySelector('.spinner-overlay');
        if (spinner) {
            spinner.remove();
        }
    }
}

// Add CSS for spinner overlay
const style = document.createElement('style');
style.textContent = `
    .spinner-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    
    .spinner-content {
        background: white;
        padding: 3rem;
        border-radius: 15px;
        text-align: center;
    }
`;
document.head.appendChild(style);

// Export functions for use in other scripts
window.apiCall = apiCall;
window.makePrediction = makePrediction;
window.getModelInfo = getModelInfo;
window.showFlashMessage = showFlashMessage;
window.formatCurrency = formatCurrency;
window.Toast = Toast;
window.Loader = Loader;
