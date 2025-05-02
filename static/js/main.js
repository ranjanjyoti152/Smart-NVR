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
    // Get the existing sidebar toggle button
    const sidebarToggle = document.getElementById('sidebar-toggle');
    
    if (sidebarToggle) {
        // Remove existing event listeners to prevent duplicates
        const newSidebarToggle = sidebarToggle.cloneNode(true);
        sidebarToggle.parentNode.replaceChild(newSidebarToggle, sidebarToggle);
        
        // Add event listener to the fresh button
        newSidebarToggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation(); // Prevent event bubbling
            document.body.classList.toggle('show-sidebar');
            console.log('Sidebar toggle clicked, show-sidebar class:', document.body.classList.contains('show-sidebar'));
        });
        
        // Add touchstart event for better mobile responsiveness
        newSidebarToggle.addEventListener('touchstart', function(e) {
            e.preventDefault();
            document.body.classList.toggle('show-sidebar');
            console.log('Sidebar toggle touched, show-sidebar class:', document.body.classList.contains('show-sidebar'));
        }, { passive: false });
    }
    
    // Close sidebar when clicking/touching outside
    document.addEventListener('click', function(e) {
        if (
            document.body.classList.contains('show-sidebar') && 
            !e.target.closest('.mac-sidebar') && 
            !e.target.closest('#sidebar-toggle')
        ) {
            document.body.classList.remove('show-sidebar');
        }
    });
    
    // Close sidebar when escape key is pressed
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && document.body.classList.contains('show-sidebar')) {
            document.body.classList.remove('show-sidebar');
        }
    });
    
    // Add touch event to close sidebar when swiping left
    const sidebar = document.querySelector('.mac-sidebar');
    if (sidebar) {
        let touchStartX = 0;
        let touchEndX = 0;
        
        sidebar.addEventListener('touchstart', function(e) {
            touchStartX = e.changedTouches[0].screenX;
        }, { passive: true });
        
        sidebar.addEventListener('touchend', function(e) {
            touchEndX = e.changedTouches[0].screenX;
            // If swiped left (start > end) by at least 50px
            if (touchStartX - touchEndX > 50) {
                document.body.classList.remove('show-sidebar');
            }
        }, { passive: true });
    }
    
    // Ensure the function runs on resize events
    window.addEventListener('resize', function() {
        // If screen size changes to desktop, hide the mobile sidebar
        if (window.innerWidth >= 992 && document.body.classList.contains('show-sidebar')) {
            document.body.classList.remove('show-sidebar');
        }
    });
}

// Add hover effects to menu items
function enhanceMenuItems() {
    const menuItems = document.querySelectorAll('.mac-menu a:not(.active)');
    menuItems.forEach(item => {
        // Add hover shine effect (CSS handles the main hover style)
        item.addEventListener('mouseover', function() {
            this.style.transition = 'all 0.3s ease'; // Ensure transition is set if needed
        });
    });
}

// Dashboard Widget Functionality
// ----------------------------

// Initialize the dashboard widgets with collapsible and drag-drop functionality
function initDashboardWidgets() {
    const widgets = document.querySelectorAll('.widget');
    const dashboardWidgets = document.getElementById('dashboard-widgets');
    const toggleWidgetsBtn = document.getElementById('toggle-widgets');
    
    if (!widgets.length || !dashboardWidgets) return;
    
    // Load widget states from localStorage
    loadWidgetStates();
    
    // Widget toggle functionality
    if (toggleWidgetsBtn) {
        toggleWidgetsBtn.addEventListener('click', function() {
            const isHidden = dashboardWidgets.classList.contains('d-none');
            
            if (isHidden) {
                dashboardWidgets.classList.remove('d-none');
                localStorage.setItem('widgets-visible', 'true');
            } else {
                dashboardWidgets.classList.add('d-none');
                localStorage.setItem('widgets-visible', 'false');
            }
        });
        
        // Check saved preference on page load
        if (localStorage.getItem('widgets-visible') === 'false') {
            dashboardWidgets.classList.add('d-none');
        }
    }
    
    // Set up collapsible functionality for each widget
    widgets.forEach(widget => {
        const header = widget.querySelector('.widget-header');
        const body = widget.querySelector('.widget-body');
        const widgetId = widget.dataset.widgetId;
        const chevron = header ? header.querySelector('i.fa-chevron-down') : null; // Get chevron icon

        if (header && body) {
            // Add click event to header for collapsing
            header.addEventListener('click', function() {
                const isCollapsed = body.classList.toggle('collapsed');
                header.classList.toggle('collapsed');
                
                // Save state to localStorage
                localStorage.setItem(`widget-${widgetId}-collapsed`, isCollapsed);

                // Optional: Add a class for animation trigger if needed, though CSS handles rotation
                if (chevron) {
                    // The CSS transition on the icon should handle the rotation smoothly
                }
            });
        }
        
        // Set up drag-and-drop functionality for rearranging widgets
        if (widget.getAttribute('draggable') === 'true') {
            widget.addEventListener('dragstart', handleDragStart);
            widget.addEventListener('dragend', handleDragEnd);
            widget.addEventListener('dragover', handleDragOver);
            widget.addEventListener('dragenter', handleDragEnter);
            widget.addEventListener('dragleave', handleDragLeave);
            widget.addEventListener('drop', handleDrop);
        }
    });
    
    // Initialize widgets with data
    initCameraHealthWidget();
    initDetectionSummaryWidget();
    initSystemResourcesWidget();
    initAIModelsWidget();
}

// Load saved widget states (collapsed/expanded) from localStorage
function loadWidgetStates() {
    const widgets = document.querySelectorAll('.widget');
    
    widgets.forEach(widget => {
        const header = widget.querySelector('.widget-header');
        const body = widget.querySelector('.widget-body');
        const widgetId = widget.dataset.widgetId;
        
        if (header && body && widgetId) {
            const isCollapsed = localStorage.getItem(`widget-${widgetId}-collapsed`) === 'true';
            
            if (isCollapsed) {
                header.classList.add('collapsed');
                body.classList.add('collapsed');
            }
        }
    });
    
    // Also check for saved positions
    const savedLayout = localStorage.getItem('widget-layout');
    if (savedLayout) {
        try {
            const layoutOrder = JSON.parse(savedLayout);
            const dashboardWidgets = document.getElementById('dashboard-widgets');
            
            if (dashboardWidgets && Array.isArray(layoutOrder)) {
                // Rearrange widgets based on saved order
                layoutOrder.forEach(id => {
                    const widget = document.querySelector(`.widget[data-widget-id="${id}"]`);
                    if (widget) {
                        dashboardWidgets.appendChild(widget);
                    }
                });
            }
        } catch (e) {
            console.error('Error restoring widget layout:', e);
        }
    }
}

// Drag and drop functionality for widgets
let draggedWidget = null;

function handleDragStart(e) {
    draggedWidget = this;
    this.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', this.dataset.widgetId);
}

function handleDragEnd(e) {
    this.classList.remove('dragging');
    document.querySelectorAll('.widget').forEach(widget => {
        widget.classList.remove('widget-drop-area');
    });
    draggedWidget = null;
    
    // Save the new widget order
    saveWidgetOrder();
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
}

function handleDragEnter(e) {
    e.preventDefault();
    if (this !== draggedWidget) {
        this.classList.add('widget-drop-area');
    }
}

function handleDragLeave(e) {
    this.classList.remove('widget-drop-area');
}

function handleDrop(e) {
    e.preventDefault();
    if (this !== draggedWidget) {
        const dashboardWidgets = document.getElementById('dashboard-widgets');
        const draggingIndex = Array.from(dashboardWidgets.children).indexOf(draggedWidget);
        const targetIndex = Array.from(dashboardWidgets.children).indexOf(this);
        
        if (draggingIndex > targetIndex) {
            dashboardWidgets.insertBefore(draggedWidget, this);
        } else {
            dashboardWidgets.insertBefore(draggedWidget, this.nextSibling);
        }
        
        this.classList.remove('widget-drop-area');
        saveWidgetOrder();
    }
}

function saveWidgetOrder() {
    const dashboardWidgets = document.getElementById('dashboard-widgets');
    if (!dashboardWidgets) return;
    
    // Get the current order of widget IDs
    const widgetOrder = Array.from(dashboardWidgets.querySelectorAll('.widget')).map(widget => widget.dataset.widgetId);
    
    // Save to localStorage
    localStorage.setItem('widget-layout', JSON.stringify(widgetOrder));
}

// Initialize Camera Health Widget
function initCameraHealthWidget() {
    const cameraHealthList = document.getElementById('camera-health-list');
    if (!cameraHealthList) return;
    
    // Clear loading placeholder
    cameraHealthList.innerHTML = '';
    
    // Get camera data
    fetch('/api/cameras')
        .then(response => response.json())
        .then(data => {
            // Check if the data has a 'cameras' property (API returns {success: true, cameras: [...]} format)
            const cameras = data.cameras || data;
            
            if (!cameras || cameras.length === 0) {
                cameraHealthList.innerHTML = '<li class="camera-status-item"><div>No cameras configured</div></li>';
                return;
            }
            
            // Add each camera to the health monitor
            cameras.forEach(camera => {
                const li = document.createElement('li');
                li.className = 'camera-status-item';
                li.dataset.cameraId = camera.id;
                
                const statusDiv = document.createElement('div');
                const statusIndicator = document.createElement('span');
                statusIndicator.className = 'status-indicator';
                statusDiv.appendChild(statusIndicator);
                statusDiv.appendChild(document.createTextNode(` ${camera.name}`));
                
                const statusText = document.createElement('span');
                statusText.className = 'camera-health-status';
                statusText.textContent = 'Checking...';
                
                li.appendChild(statusDiv);
                li.appendChild(statusText);
                cameraHealthList.appendChild(li);
                
                // Check camera status
                checkCameraStatus(camera.id);
            });
            
            // Set up periodic status updates
            setInterval(updateAllCameraStatuses, 30000); // Update every 30 seconds
        })
        .catch(error => {
            console.error('Error fetching cameras:', error);
            cameraHealthList.innerHTML = '<li class="camera-status-item"><div>Error loading camera data</div></li>';
        });
}

// Check individual camera status
function checkCameraStatus(cameraId) {
    const statusItem = document.querySelector(`.camera-status-item[data-camera-id="${cameraId}"]`);
    if (!statusItem) return;
    
    const statusIndicator = statusItem.querySelector('.status-indicator');
    const statusText = statusItem.querySelector('.camera-health-status');
    
    fetch(`/api/cameras/${cameraId}/status`)
        .then(response => response.json())
        .then(status => {
            if (status.online) {
                statusIndicator.className = 'status-indicator online';
                statusText.textContent = 'Online';
            } else {
                statusIndicator.className = 'status-indicator offline';
                statusText.textContent = 'Offline';
            }
            
            if (status.warn) {
                statusIndicator.className = 'status-indicator warning';
                statusText.textContent = status.warn_reason || 'Warning';
            }
        })
        .catch(error => {
            console.error(`Error checking camera ${cameraId} status:`, error);
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'Error';
        });
}

// Update all camera statuses
function updateAllCameraStatuses() {
    const cameraItems = document.querySelectorAll('.camera-status-item[data-camera-id]');
    cameraItems.forEach(item => {
        const cameraId = item.dataset.cameraId;
        checkCameraStatus(cameraId);
    });
}

// Initialize Detection Summary Widget
function initDetectionSummaryWidget() {
    const detectionList = document.getElementById('detection-list');
    const chartContainer = document.getElementById('detection-chart');
    
    if (!detectionList || !chartContainer) return;
    
    // Clear loading placeholder
    detectionList.innerHTML = '';
    
    // Fetch detection summary
    fetch('/api/detections/summary')
        .then(response => response.json())
        .then(data => {
            if (!data || !data.classes || Object.keys(data.classes).length === 0) {
                detectionList.innerHTML = '<div class="detection-item"><span>No detections in the last 24 hours</span></div>';
                return;
            }
            
            // Sort classes by count (descending)
            const sortedClasses = Object.entries(data.classes)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10); // Top 10 classes
            
            // Set up chart data
            const chartLabels = sortedClasses.map(item => item[0]);
            const chartValues = sortedClasses.map(item => item[1]);
            
            // Create chart using Chart.js (if available)
            if (window.Chart) {
                new Chart(chartContainer, {
                    type: 'bar',
                    data: {
                        labels: chartLabels,
                        datasets: [{
                            label: 'Detections',
                            data: chartValues,
                            backgroundColor: 'rgba(75, 192, 192, 0.5)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                precision: 0
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            }
                        }
                    }
                });
            }
            
            // Add detection counts to list
            sortedClasses.forEach(([className, count]) => {
                const item = document.createElement('div');
                item.className = 'detection-item';
                
                const nameSpan = document.createElement('span');
                nameSpan.textContent = className;
                
                const countSpan = document.createElement('span');
                countSpan.className = 'detection-count';
                countSpan.textContent = count;
                
                item.appendChild(nameSpan);
                item.appendChild(countSpan);
                detectionList.appendChild(item);
            });
        })
        .catch(error => {
            console.error('Error fetching detection summary:', error);
            detectionList.innerHTML = '<div class="detection-item"><span>Error loading detection data</span></div>';
        });
}

// Initialize System Resources Widget
function initSystemResourcesWidget() {
    // Get references to resource progress bars
    const cpuBar = document.getElementById('cpu-bar');
    const cpuUsage = document.getElementById('cpu-usage');
    const memoryBar = document.getElementById('memory-bar');
    const memoryUsage = document.getElementById('memory-usage');
    const diskBar = document.getElementById('disk-bar');
    const diskUsage = document.getElementById('disk-usage');
    const gpuResources = document.getElementById('gpu-resources');
    
    if (!cpuBar || !memoryBar || !diskBar || !gpuResources) return;
    
    // Function to update system resources
    function updateSystemResources() {
        fetch('/api/system/resources')
            .then(response => response.json())
            .then(data => {
                // Extract resources from response
                const resources = data.resources || data;
                
                // Update CPU
                if (resources.cpu) {
                    const cpuPercent = Math.round(resources.cpu.percent || resources.cpu.usage_percent || 0);
                    cpuBar.style.width = `${cpuPercent}%`;
                    cpuUsage.textContent = `${cpuPercent}%`;
                }
                
                // Update Memory
                if (resources.memory) {
                    const memPercent = Math.round(resources.memory.percent || resources.memory.used_percent || 0);
                    memoryBar.style.width = `${memPercent}%`;
                    memoryUsage.textContent = `${memPercent}% (${formatMemorySize(resources.memory.used)} / ${formatMemorySize(resources.memory.total)})`;
                }
                
                // Update Disk
                if (resources.disk) {
                    const diskPercent = Math.round(resources.disk.percent || resources.disk.used_percent || 0);
                    diskBar.style.width = `${diskPercent}%`;
                    diskUsage.textContent = `${diskPercent}% (${formatMemorySize(resources.disk.used)} / ${formatMemorySize(resources.disk.total)})`;
                }
                
                // Update GPU(s) if available
                if (resources.gpu && resources.gpu.length > 0) {
                    gpuResources.innerHTML = ''; // Clear existing GPU items
                    
                    resources.gpu.forEach((gpu, index) => {
                        const gpuItem = document.createElement('div');
                        gpuItem.className = 'resource-item';
                        
                        const gpuLabel = document.createElement('div');
                        gpuLabel.className = 'resource-label';
                        const gpuUtilization = gpu.utilization || gpu.load || 0;
                        gpuLabel.innerHTML = `<span>GPU ${index + 1} - ${gpu.name || 'Unknown'}</span><span id="gpu-${index}-usage">${gpuUtilization}%</span>`;
                        
                        const gpuProgress = document.createElement('div');
                        gpuProgress.className = 'resource-progress';
                        
                        const gpuBar = document.createElement('div');
                        gpuBar.className = 'resource-bar gpu';
                        gpuBar.style.width = `${gpuUtilization}%`;
                        
                        gpuProgress.appendChild(gpuBar);
                        gpuItem.appendChild(gpuLabel);
                        gpuItem.appendChild(gpuProgress);
                        gpuResources.appendChild(gpuItem);
                        
                        // Add memory info if available
                        if (gpu.memory) {
                            const memPercent = gpu.memory_percent || Math.round((gpu.memory.used / gpu.memory.total) * 100) || 0;
                            
                            const memItem = document.createElement('div');
                            memItem.className = 'resource-item gpu-memory-item';
                            
                            const memLabel = document.createElement('div');
                            memLabel.className = 'resource-label';
                            memLabel.innerHTML = `<span>GPU ${index + 1} Memory</span><span>${memPercent}% (${formatMemorySize(gpu.memory.used)} / ${formatMemorySize(gpu.memory.total)})</span>`;
                            
                            const memProgress = document.createElement('div');
                            memProgress.className = 'resource-progress';
                            
                            const memBar = document.createElement('div');
                            memBar.className = 'resource-bar gpu';
                            memBar.style.width = `${memPercent}%`;
                            
                            memProgress.appendChild(memBar);
                            memItem.appendChild(memLabel);
                            memItem.appendChild(memProgress);
                            gpuResources.appendChild(memItem);
                        }
                    });
                } else {
                    gpuResources.innerHTML = '<div class="text-muted small">No GPU detected</div>';
                }
            })
            .catch(error => {
                console.error('Error updating system resources:', error);
                // Don't update UI on error to preserve last known values
            });
    }
    
    // Helper function to format memory size
    function formatMemorySize(bytes) {
        if (!bytes) return '0 B';
        
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return `${(bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0)} ${sizes[i]}`;
    }
    
    // Initial update
    updateSystemResources();
    
    // Update resources every 5 seconds
    setInterval(updateSystemResources, 5000);
}

// Initialize AI Model Status Widget
function initAIModelsWidget() {
    const modelList = document.getElementById('model-list');
    if (!modelList) return;
    
    // Clear loading placeholder
    modelList.innerHTML = '';
    
    // Fetch AI models
    fetch('/api/ai/models')
        .then(response => response.json())
        .then(models => {
            if (!models || models.length === 0) {
                modelList.innerHTML = '<div class="model-item"><div class="model-info"><span class="model-name">No AI models configured</span></div></div>';
                return;
            }
            
            // Add each model to the list
            models.forEach(model => {
                const item = document.createElement('div');
                item.className = 'model-item';
                
                const info = document.createElement('div');
                info.className = 'model-info';
                
                const name = document.createElement('span');
                name.className = 'model-name';
                name.textContent = model.name || 'Unknown Model';
                
                const details = document.createElement('span');
                details.className = 'model-details';
                details.textContent = `${model.framework || 'Unknown'} | ${model.version || 'Unknown Version'}`;
                
                const badge = document.createElement('span');
                badge.className = 'model-badge';
                badge.textContent = model.status || 'Unknown';
                
                // Apply special class for active model
                if (model.is_active) {
                    badge.classList.add('active-model');
                    badge.textContent = 'Active';
                }
                
                info.appendChild(name);
                info.appendChild(details);
                item.appendChild(info);
                item.appendChild(badge);
                modelList.appendChild(item);
            });
        })
        .catch(error => {
            console.error('Error fetching AI models:', error);
            modelList.innerHTML = '<div class="model-item"><div class="model-info"><span class="model-name">Error loading models</span></div></div>';
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
    
    // Initialize dashboard widgets if on dashboard page
    if (document.getElementById('dashboard-widgets')) {
        initDashboardWidgets();
    }
});

// Dark mode toggle with enhanced animation
function toggleDarkMode() {
    const currentMode = localStorage.getItem('darkMode') === 'true';
    localStorage.setItem('darkMode', !currentMode);
    
    const root = document.documentElement;
    const toggleButton = document.querySelector('.dark-mode-toggle'); // Get the button itself
    
    // Add a class to the body for global transition effects
    document.body.classList.add('theme-transitioning');

    if (!currentMode) {
        // Add transition for smooth color changes
        root.style.transition = 'all 0.5s ease';
        root.setAttribute('data-theme', 'dark');
        
        // Change dark mode button icon and add animation
        const darkModeToggleIcon = document.querySelector('.dark-mode-toggle i');
        if (darkModeToggleIcon) {
            darkModeToggleIcon.className = 'fas fa-sun';
        }
        if (toggleButton) {
            toggleButton.classList.add('toggled'); // Add class for potential animation
        }
        
        showToast('Dark mode activated ✨', 'info', 2000);
    } else {
        root.style.transition = 'all 0.5s ease';
        root.removeAttribute('data-theme');
        
        // Change dark mode button icon back to moon and add animation
        const darkModeToggleIcon = document.querySelector('.dark-mode-toggle i');
        if (darkModeToggleIcon) {
            darkModeToggleIcon.className = 'fas fa-moon';
        }
         if (toggleButton) {
            toggleButton.classList.remove('toggled'); // Remove class
        }
        
        showToast('Light mode activated ☀️', 'info', 2000);
    }
    
    // Add animation to all cards during mode change
    const cards = document.querySelectorAll('.mac-card');
    cards.forEach(card => {
        card.style.transition = 'all 0.5s ease';
        card.classList.add('pulse-animation');
        setTimeout(() => {
            card.classList.remove('pulse-animation');
            // Remove inline transition style after animation to avoid conflicts
            card.style.transition = ''; 
        }, 1000);
    });

    // Remove the global transition class after the transition duration
    setTimeout(() => {
        document.body.classList.remove('theme-transitioning');
        root.style.transition = ''; // Remove inline style to rely on CSS
    }, 500); // Match CSS transition duration
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
    }
}

// Run dark mode check on page load
checkDarkMode();