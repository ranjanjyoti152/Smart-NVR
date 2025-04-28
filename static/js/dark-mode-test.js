/**
 * Smart-NVR Dark Mode Test Script
 * This script helps identify elements that aren't properly styled in dark mode
 */

document.addEventListener('DOMContentLoaded', function() {
    // Create a button to highlight elements that might need dark mode styling
    const createTestButton = () => {
        const buttonContainer = document.createElement('div');
        buttonContainer.style.position = 'fixed';
        buttonContainer.style.bottom = '80px';
        buttonContainer.style.right = '20px';
        buttonContainer.style.zIndex = '9999';
        
        const button = document.createElement('button');
        button.textContent = 'Test Dark Mode';
        button.style.padding = '8px 12px';
        button.style.borderRadius = '8px';
        button.style.backgroundColor = 'var(--color-primary)';
        button.style.color = 'white';
        button.style.border = 'none';
        button.style.cursor = 'pointer';
        
        button.addEventListener('click', highlightNonDarkModeElements);
        buttonContainer.appendChild(button);
        document.body.appendChild(buttonContainer);
    };
    
    // Function to identify and highlight elements that might not be properly styled for dark mode
    const highlightNonDarkModeElements = () => {
        // Common selectors that might need dark mode styling
        const selectors = [
            'input',
            'select',
            'textarea',
            '.form-control',
            '.form-select',
            '.card',
            '.table',
            '.modal-content',
            '.dropdown-menu',
            '.dropdown-item',
            '.nav-tabs .nav-link',
            '.alert',
            '.badge',
            '.btn-outline-secondary',
            '.btn-outline-dark',
            '.list-group-item'
        ];
        
        // Check if we're in dark mode
        const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
        if (!isDarkMode) {
            alert('Please enable dark mode first to test dark mode compatibility');
            return;
        }
        
        // Get all potentially problematic elements
        let elementsToCheck = [];
        selectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(el => elementsToCheck.push(el));
        });
        
        // Function to get computed background color
        const getBackgroundColor = (element) => {
            const style = window.getComputedStyle(element);
            return style.backgroundColor;
        };
        
        // Check which elements might have improper styling
        elementsToCheck.forEach(element => {
            const bgColor = getBackgroundColor(element);
            
            // Check if the background color is likely default white or very light
            if (
                bgColor === 'rgb(255, 255, 255)' || 
                bgColor === '#ffffff' || 
                bgColor === 'white' ||
                (bgColor.startsWith('rgba') && bgColor.includes('1)') && !bgColor.match(/rgba\(\s*0\s*,\s*0\s*,\s*0\s*,/))
            ) {
                // Highlight the element
                element.style.outline = '2px solid red';
                element.style.position = 'relative';
                
                // Add a tooltip
                const tooltip = document.createElement('div');
                tooltip.textContent = 'Possible dark mode styling issue';
                tooltip.style.position = 'absolute';
                tooltip.style.top = '0';
                tooltip.style.left = '0';
                tooltip.style.backgroundColor = 'red';
                tooltip.style.color = 'white';
                tooltip.style.padding = '2px 5px';
                tooltip.style.fontSize = '10px';
                tooltip.style.borderRadius = '3px';
                tooltip.style.zIndex = '9999';
                tooltip.style.pointerEvents = 'none';
                
                element.appendChild(tooltip);
                
                // Add to console for debugging
                console.warn('Possible dark mode styling issue:', element);
            }
        });
        
        console.log('Dark mode test complete. Elements with potential issues are highlighted in red and logged to console.');
    };
    
    // Create the test button
    createTestButton();
    
    // Add dark mode info to the console
    console.info(
        'Dark mode testing utilities loaded. Use the "Test Dark Mode" button to identify elements ' +
        'that might not be properly styled in dark mode.'
    );
});