// Form validation and dynamic interactions
document.addEventListener('DOMContentLoaded', function() {
    // Input validation for numerical fields
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('input', function() {
            validateNumericInput(this);
        });
    });

    // Form submission handling
    const predictionForm = document.querySelector('.prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            if (!validateForm(this)) {
                e.preventDefault();
            }
        });
    }

    // Calculator form handling
    const calculatorForm = document.querySelector('.calculator-form');
    if (calculatorForm) {
        calculatorForm.addEventListener('submit', function(e) {
            if (!validateCalculatorForm(this)) {
                e.preventDefault();
            }
        });
    }

    // Dynamic value updates for calculator
    setupCalculatorListeners();
});

// Validate numeric input fields
function validateNumericInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.getAttribute('min'));
    const max = parseFloat(input.getAttribute('max'));

    if (isNaN(value)) {
        showError(input, 'Please enter a valid number');
        return false;
    }

    if (min !== null && value < min) {
        showError(input, `Value must be at least ${min}`);
        return false;
    }

    if (max !== null && value > max) {
        showError(input, `Value must be less than ${max}`);
        return false;
    }

    clearError(input);
    return true;
}

// Form validation
function validateForm(form) {
    let isValid = true;
    const inputs = form.querySelectorAll('input, select');

    inputs.forEach(input => {
        if (input.hasAttribute('required') && !input.value.trim()) {
            showError(input, 'This field is required');
            isValid = false;
        } else if (input.type === 'number') {
            isValid = validateNumericInput(input) && isValid;
        }
    });

    return isValid;
}

// Calculator form validation
function validateCalculatorForm(form) {
    let isValid = true;
    const inputs = form.querySelectorAll('input[type="number"]');

    inputs.forEach(input => {
        const value = parseFloat(input.value);
        if (isNaN(value) || value < 0) {
            showError(input, 'Please enter a valid positive number');
            isValid = false;
        }
    });

    return isValid;
}

// Setup calculator input listeners
function setupCalculatorListeners() {
    const costInputs = document.querySelectorAll('.calculator-form input[type="number"]');
    costInputs.forEach(input => {
        input.addEventListener('input', updateTotalCost);
    });
}

// Update total cost calculation
function updateTotalCost() {
    const costInputs = document.querySelectorAll('.calculator-form input[type="number"]');
    let total = 0;

    costInputs.forEach(input => {
        const value = parseFloat(input.value) || 0;
        total += value;
    });

    const totalDisplay = document.getElementById('total-cost');
    if (totalDisplay) {
        totalDisplay.textContent = `₹${total.toLocaleString('en-IN', {
            maximumFractionDigits: 2,
            minimumFractionDigits: 2
        })}`;
    }
}

// Error display functions
function showError(input, message) {
    clearError(input);
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    input.parentNode.appendChild(errorDiv);
    input.classList.add('error');
}

function clearError(input) {
    const errorDiv = input.parentNode.querySelector('.error-message');
    if (errorDiv) {
        errorDiv.remove();
    }
    input.classList.remove('error');
}

// Animation for results
function animateResults() {
    const results = document.querySelectorAll('.crop-card, .revenue-card');
    results.forEach((element, index) => {
        element.style.animationDelay = `${index * 0.2}s`;
    });
}

// Initialize animations
if (document.querySelector('.results-section')) {
    animateResults();
}

// Handle chart resizing
window.addEventListener('resize', function() {
    const pieChart = document.getElementById('pieChart');
    if (pieChart && pieChart.data) {
        Plotly.Plots.resize(pieChart);
    }
});
