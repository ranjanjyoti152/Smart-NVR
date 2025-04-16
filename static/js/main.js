/**
 * SmartNVR - Main JavaScript Functions
 * Contains utility functions for enhancing the UI experience
 */

// Create a loading spinner with optional text
function createLoader(container, size = 'normal', text = null) {
    // Clear any existing content
    if (typeof container === 'string') {
        container = document.querySelector(container);
    }
    
    if (!container) return;
    
    container.innerHTML = '';
    
    const loaderContainer = document.createElement('div');
    loaderContainer.className = 'mac-loader-container';
    
    const loader = document.createElement('div');
    loader.className = size === 'small' ? 'mac-loader sm' : 'mac-loader';
    loaderContainer.appendChild(loader);
    
    if (text) {
        const textElement = document.createElement('div');
        textElement.className = 'mac-loader-text';
        textElement.textContent = text;
        loaderContainer.appendChild(textElement);
    }
    
    container.appendChild(loaderContainer);
    return loaderContainer;
}

// Create a toast notification
function showToast(message, type = 'info', duration = 3000) {
    // Remove existing toasts
    const existingToasts = document.querySelectorAll('.mac-toast');
    existingToasts.forEach(toast => {
        toast.classList.remove('mac-toast-show');
        setTimeout(() => toast.remove(), 300);
    });
    
    // Create new toast
    const toast = document.createElement('div');
    toast.className = `mac-toast mac-toast-${type}`;
    
    // Add icon based on type
    let icon = 'info-circle';
    if (type === 'success') icon = 'check-circle';
    if (type === 'warning') icon = 'exclamation-triangle';
    if (type === 'error') icon = 'exclamation-circle';
    
    toast.innerHTML = `
        <div class="mac-toast-content">
            <i class="fas fa-${icon} me-2"></i>
            <span>${message}</span>
        </div>
        <button class="mac-toast-close">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    document.body.appendChild(toast);
    
    // Show the toast with animation
    setTimeout(() => toast.classList.add('mac-toast-show'), 10);
    
    // Add close button functionality
    const closeBtn = toast.querySelector('.mac-toast-close');
    closeBtn.addEventListener('click', () => {
        toast.classList.remove('mac-toast-show');
        setTimeout(() => toast.remove(), 300);
    });
    
    // Auto-dismiss after duration
    if (duration) {
        setTimeout(() => {
            if (document.body.contains(toast)) {
                toast.classList.remove('mac-toast-show');
                setTimeout(() => toast.remove(), 300);
            }
        }, duration);
    }
    
    return toast;
}

// Add smooth animations to cards when they appear
function animateCardsOnScroll() {
    const cards = document.querySelectorAll('.mac-card');
    
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('mac-card-visible');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1
        });
        
        cards.forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            observer.observe(card);
        });
    } else {
        // Fallback for browsers without IntersectionObserver
        cards.forEach(card => card.classList.add('mac-card-visible'));
    }
}

// Initialize sidebar toggle functionality for mobile
function initSidebarToggle() {
    // Add hamburger menu button if it doesn't exist already
    const header = document.querySelector('.mac-header .container-fluid .row');
    
    if (header && window.innerWidth < 992 && !document.querySelector('#sidebar-toggle')) {
        const toggleCol = document.createElement('div');
        toggleCol.className = 'col-auto d-lg-none';
        
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'btn btn-link text-decoration-none p-0 mac-sidebar-toggle';
        toggleBtn.id = 'sidebar-toggle';
        toggleBtn.innerHTML = '<i class="fas fa-bars"></i>';
        
        toggleCol.appendChild(toggleBtn);
        header.prepend(toggleCol);
        
        // Add event listener
        toggleBtn.addEventListener('click', () => {
            document.body.classList.toggle('show-sidebar');
        });
        
        // Close sidebar when clicking outside
        document.addEventListener('click', (e) => {
            if (
                document.body.classList.contains('show-sidebar') && 
                !e.target.closest('.mac-sidebar') && 
                !e.target.closest('#sidebar-toggle')
            ) {
                document.body.classList.remove('show-sidebar');
            }
        });
    }
}

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Animation for cards
    animateCardsOnScroll();
    
    // Mobile sidebar toggle
    initSidebarToggle();
    
    // Add animation class to body after page loads
    document.body.classList.add('page-loaded');
});

// Dark mode toggle
function toggleDarkMode() {
    const currentMode = localStorage.getItem('darkMode') === 'true';
    localStorage.setItem('darkMode', !currentMode);
    
    if (!currentMode) {
        document.documentElement.setAttribute('data-theme', 'dark');
        showToast('Dark mode enabled', 'info', 2000);
    } else {
        document.documentElement.removeAttribute('data-theme');
        showToast('Light mode enabled', 'info', 2000);
    }
}

// Check for dark mode on page load
function checkDarkMode() {
    // Check localStorage preference
    if (localStorage.getItem('darkMode') === 'true') {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
    // No else needed as we want to respect system preference by default
}

// Run dark mode check on page load
checkDarkMode();