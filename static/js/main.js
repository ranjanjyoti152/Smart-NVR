/**
 * SmartNVR - Enhanced JavaScript Functions
 * Contains utility functions for a colorful and attractive UI experience
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

// Create a toast notification with enhanced styling
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
            <i class="fas fa-${icon}"></i>
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
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });
        
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            // Add staggered delay for card animations
            card.style.transitionDelay = `${index * 0.05}s`;
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

// Add hover effects to menu items
function enhanceMenuItems() {
    const menuItems = document.querySelectorAll('.mac-menu a:not(.active)');
    menuItems.forEach(item => {
        // Add hover shine effect
        item.addEventListener('mouseover', function() {
            this.style.transition = 'all 0.3s ease';
        });
        
        item.addEventListener('mousemove', function(e) {
            const x = e.pageX - this.offsetLeft;
            const y = e.pageY - this.offsetTop;
            
            this.style.background = `radial-gradient(circle at ${x}px ${y}px, rgba(var(--color-primary-rgb, 75, 109, 222), 0.2) 0%, rgba(var(--color-primary-rgb, 75, 109, 222), 0.1) 20%, rgba(var(--color-primary-rgb, 75, 109, 222), 0.05) 60%)`;
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.background = '';
        });
    });
}

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Animation for cards
    animateCardsOnScroll();
    
    // Mobile sidebar toggle
    initSidebarToggle();
    
    // Add animation class to body after page loads
    document.body.classList.add('page-loaded');
    
    // Add hover effects to menu items
    enhanceMenuItems();
    
    // Add stat number animation on hover
    const statNumbers = document.querySelectorAll('.stat-number');
    statNumbers.forEach(stat => {
        stat.addEventListener('mouseenter', function() {
            this.classList.add('pulse-animation');
        });
        
        stat.addEventListener('mouseleave', function() {
            this.classList.remove('pulse-animation');
        });
    });
});

// Dark mode toggle with enhanced animation
function toggleDarkMode() {
    const currentMode = localStorage.getItem('darkMode') === 'true';
    localStorage.setItem('darkMode', !currentMode);
    
    const root = document.documentElement;
    
    if (!currentMode) {
        // Add transition for smooth color changes
        root.style.transition = 'all 0.5s ease';
        root.setAttribute('data-theme', 'dark');
        
        // Change dark mode button icon
        const darkModeToggle = document.querySelector('.dark-mode-toggle i');
        if (darkModeToggle) {
            darkModeToggle.className = 'fas fa-sun';
        }
        
        showToast('Dark mode activated ✨', 'info', 2000);
    } else {
        root.style.transition = 'all 0.5s ease';
        root.removeAttribute('data-theme');
        
        // Change dark mode button icon back to moon
        const darkModeToggle = document.querySelector('.dark-mode-toggle i');
        if (darkModeToggle) {
            darkModeToggle.className = 'fas fa-moon';
        }
        
        showToast('Light mode activated ☀️', 'info', 2000);
    }
    
    // Fix for form controls in dark mode
    const formControls = document.querySelectorAll('.form-control, .form-select');
    formControls.forEach(control => {
        control.style.transition = 'all 0.5s ease';
        // Force input elements to respect theme changes
        if (!currentMode) {
            control.classList.add('dark-mode-input');
        } else {
            control.classList.remove('dark-mode-input');
        }
    });
    
    // Add animation to all cards during mode change
    const cards = document.querySelectorAll('.mac-card');
    cards.forEach(card => {
        card.style.transition = 'all 0.5s ease';
        card.classList.add('pulse-animation');
        setTimeout(() => {
            card.classList.remove('pulse-animation');
        }, 1000);
    });
}

// Check for dark mode on page load
function checkDarkMode() {
    // Check localStorage preference
    if (localStorage.getItem('darkMode') === 'true') {
        document.documentElement.setAttribute('data-theme', 'dark');
        
        // Set the correct icon for the dark mode toggle button
        const darkModeToggle = document.querySelector('.dark-mode-toggle i');
        if (darkModeToggle) {
            darkModeToggle.className = 'fas fa-sun';
        }
        
        // Apply dark mode to form elements
        const formControls = document.querySelectorAll('.form-control, .form-select');
        formControls.forEach(control => {
            control.classList.add('dark-mode-input');
        });
    }
}

// Run dark mode check on page load
checkDarkMode();