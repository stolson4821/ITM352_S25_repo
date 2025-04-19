document.addEventListener('DOMContentLoaded', function() {
    // Check if current page should have cherry blossoms
    const currentPage = window.location.pathname.split('/').pop();
    const enabledPages = [
      'index.html', 
      'questions.html', 
      'ready_to_begin.html', 
      'select_difficulty.html', 
      'thank_you.html',
      '' // This catches the root URL (e.g., domain.com/ which might serve index.html)
    ];
    
    // Create theme toggle button (on all pages)
    const themeToggle = document.createElement('div');
    themeToggle.className = 'theme-toggle';
    themeToggle.innerHTML = '🌙';
    themeToggle.addEventListener('click', toggleTheme);
    document.body.appendChild(themeToggle);
    
    // Check if this is one of our cherry blossom enabled pages
    if (enabledPages.includes(currentPage)) {
      // Add cherry blossom class to body
      document.body.classList.add('cherry-blossom-enabled');
      
      // Create cherry blossoms
      createCherryBlossoms();
      
      // Add fade-in animation to elements
      const mainElements = document.querySelectorAll('form, .score-summary, .history, .leaderboard, h1, h2');
      mainElements.forEach((el, index) => {
        setTimeout(() => {
          el.classList.add('fade-in');
        }, index * 100);
      });
    }
    
    // Check for saved theme preference (on all pages)
    if (localStorage.getItem('theme') === 'dark') {
      document.body.classList.add('dark-theme');
      document.querySelector('.theme-toggle').innerHTML = '☀️';
    }
  });
  
  // Toggle between light and dark theme
  function toggleTheme() {
    const body = document.body;
    const themeToggle = document.querySelector('.theme-toggle');
    
    if (body.classList.contains('dark-theme')) {
      body.classList.remove('dark-theme');
      themeToggle.innerHTML = '🌙';
      localStorage.setItem('theme', 'light');
    } else {
      body.classList.add('dark-theme');
      themeToggle.innerHTML = '☀️';
      localStorage.setItem('theme', 'dark');
    }
  }
  
  // Create the initial set of cherry blossoms
  function createCherryBlossoms() {
    const blossomsCount = 50;
    const container = document.body;
    
    for (let i = 0; i < blossomsCount; i++) {
      setTimeout(() => {
        createSingleBlossom(container, true);
      }, i * 300); // Stagger the creation
    }
  }
  
  // Create a single cherry blossom
  function createSingleBlossom(container = document.body, isInitial = false) {
    const blossom = document.createElement('div');
    blossom.className = 'cherry-blossom';
    
    // Random blossom appearance
    const size = Math.random() * 15 + 5; // 5-20px
    const isLight = Math.random() > 0.5;
    
    blossom.style.width = `${size}px`;
    blossom.style.height = `${size}px`;
    blossom.style.backgroundColor = isLight ? 'var(--cherry-color-2)' : 'var(--cherry-color-1)';
    blossom.style.borderRadius = '50% 50% 50% 0';
    blossom.style.left = `${Math.random() * 100}vw`;
    
    // Animation properties
    const fallDuration = Math.random() * 10 + 15; // 15-25s
    const swayDuration = Math.random() * 3 + 3; // 3-6s
    
    blossom.style.animation = `falling ${fallDuration}s linear forwards, sway ${swayDuration}s ease-in-out infinite`;
    
    // If this is part of the initial batch, randomize the start position
    if (isInitial) {
      blossom.style.animationDelay = `${Math.random() * 20}s`; // Staggered start
    }
    
    blossom.style.opacity = Math.random() * 0.6 + 0.4; // 0.4-1.0
    blossom.style.transform = `rotate(${Math.random() * 360}deg)`;
    
    container.appendChild(blossom);
    
    // Remove blossom after animation completes and create a new one
    setTimeout(() => {
      // Only create a new blossom if we're still on the same page
      if (document.body.classList.contains('cherry-blossom-enabled')) {
        container.removeChild(blossom);
        createSingleBlossom(container);
      } else {
        // If page is no longer one that should have cherry blossoms, just remove it
        if (blossom.parentNode) {
          container.removeChild(blossom); 
        }
      }
    }, fallDuration * 1000);
  }