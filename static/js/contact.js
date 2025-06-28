// This is just for handling the mobile navigation toggle
document.addEventListener("DOMContentLoaded", function () {
    const mobileNavToggle = document.querySelector(".mobile-nav-toggle");
    const navMenu = document.querySelector(".navmenu");
  
    if (mobileNavToggle) {
      mobileNavToggle.addEventListener("click", function () {
        navMenu.classList.toggle("navbar-mobile");
        this.classList.toggle("bi-list");
        this.classList.toggle("bi-x");
      });
    }
  });