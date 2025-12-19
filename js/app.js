// Routing and Navigation
let currentPage = 'login-page';
let selectedRole = 'Field Officer';

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  // Set initial page based on URL hash or default to login
  const hash = window.location.hash.slice(1) || '/';
  navigateTo(hash);

  // Role selection handlers
  const roleButtons = document.querySelectorAll('.role');
  roleButtons.forEach(button => {
    button.addEventListener('click', function() {
      roleButtons.forEach(btn => btn.classList.remove('active'));
      this.classList.add('active');
      selectedRole = this.getAttribute('data-role');
    });
  });

  // Login button handler
  const loginButton = document.getElementById('login-button');
  if (loginButton) {
    loginButton.addEventListener('click', handleLogin);
  }

  // Connect CCTV button handler
  const connectBtn = document.getElementById('connect-cctv-btn');
  if (connectBtn) {
    connectBtn.addEventListener('click', handleConnectCCTV);
  }

  // Photo upload handler
  const photoUpload = document.getElementById('photo-upload');
  if (photoUpload) {
    photoUpload.addEventListener('click', function() {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.onchange = function(e) {
        const file = e.target.files[0];
        if (file) {
          photoUpload.innerHTML = `File selected: ${file.name}<small>High quality photos improve recognition accuracy</small>`;
        }
      };
      input.click();
    });
  }

  // Update active nav links
  updateActiveNavLinks();
});

// Navigation function (exposed globally for onclick handlers)
window.navigateTo = function(path) {
  // Hide all pages
  const pages = document.querySelectorAll('.page');
  pages.forEach(page => page.classList.remove('active'));

  // Map paths to page IDs
  const routeMap = {
    '/': 'login-page',
    '/post-login': 'post-login-page',
    '/dashboard': 'dashboard-page',
    '/upload-person': 'upload-person-page',
    '/alerts': 'alerts-page',
    '/movement': 'movement-page'
  };

  const pageId = routeMap[path] || 'login-page';
  const targetPage = document.getElementById(pageId);
  
  if (targetPage) {
    targetPage.classList.add('active');
    currentPage = pageId;
    
    // Update URL hash
    window.location.hash = path;
    
    // Update active nav links
    updateActiveNavLinks();
  }
};

// Update active navigation links
function updateActiveNavLinks() {
  const navLinks = document.querySelectorAll('.nav-menu a');
  const currentHash = window.location.hash.slice(1) || '/';
  
  navLinks.forEach(link => {
    link.classList.remove('active');
    const onclick = link.getAttribute('onclick');
    if (onclick) {
      const pathMatch = onclick.match(/\/[^'"]+/);
      if (pathMatch) {
        const path = pathMatch[0];
        if (currentHash === path || (currentHash === '/' && path === '/dashboard')) {
          link.classList.add('active');
        }
      }
    }
  });
  
  // Also handle pages that don't have nav (like post-login)
  if (currentHash === '/post-login') {
    navLinks.forEach(link => link.classList.remove('active'));
  }
}

// Handle login
function handleLogin() {
  // Store role in sessionStorage (or localStorage)
  sessionStorage.setItem('userRole', selectedRole);
  
  // Navigate to post-login page
  navigateTo('/post-login');
}

// Handle CCTV connection
function handleConnectCCTV() {
  const connectingBox = document.getElementById('connecting-box');
  if (!connectingBox) return;

  // Show spinner
  connectingBox.innerHTML = `
    <div class="spinner" aria-hidden></div>
    <h3>Connecting to nearby CCTV feeds…</h3>
    <p class="muted">This may take a few seconds</p>
  `;

  // Simulate connection delay then navigate to dashboard
  setTimeout(() => {
    navigateTo('/dashboard');
  }, 2200);
}

// Handle browser back/forward buttons
window.addEventListener('hashchange', function() {
  const hash = window.location.hash.slice(1) || '/';
  navigateTo(hash);
});

