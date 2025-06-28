// Global variables
let currentProfile = null;
let activeTab = "about";

// Initialize profile page
document.addEventListener("DOMContentLoaded", function () {
  loadProfile();
  initializeTabs();
  initializeImageUpload();
  initializeModals();
  initializeFormHandlers();
});

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Load profile data
async function loadProfile() {
  try {
    const response = await fetch("/api/profile");
    if (!response.ok) {
      throw new Error("Failed to load profile");
    }
    currentProfile = await response.json();
    await delay(100); // Small delay before UI update
    updateUI();
  } catch (error) {
    console.error("Error loading profile:", error);
    // Check for recent successful submission
    const successAlert = document.querySelector(".alert.alert-success");
    const isRecentSuccess =
      successAlert &&
      (successAlert.textContent.includes("Project") ||
        successAlert.textContent.includes("Extracurricular"));

    if (!isRecentSuccess) {
      showAlert("Failed to load profile data", "error");
    }
  } finally {
    const preloader = document.querySelector("#preloader");
    if (preloader) {
      preloader.style.display = "none";
    }
  }
}

// Update UI with profile data
function updateUI() {
  // Update hero section
  document.getElementById("profileName").textContent =
    currentProfile.personalInfo.name;
  document.getElementById("profileBio").textContent =
    currentProfile.personalInfo.bio || "No bio added yet";
  const profileImage = document.getElementById("profileImage");
  if (currentProfile.personalInfo.profileImage) {
    document.getElementById(
      "profileImage"
    ).src = `/static/uploads/profile_images/${currentProfile.personalInfo.profileImage}`;
  } else {
    profileImage.src = "/static/uploads/profile_images/default.jpg"; // Fallback to default image
  }

  // Update personal info
  const personalInfo = document.getElementById("personalInfo");
  personalInfo.innerHTML = `
        <div class="info-grid">
            <div class="info-item">
                <strong>Email:</strong> ${currentProfile.personalInfo.email}
            </div>
            <div class="info-item">
                <strong>Phone:</strong> ${
                  currentProfile.personalInfo.phone || "Not provided"
                }
            </div>
            <div class="info-item">
                <strong>Location:</strong> ${
                  currentProfile.personalInfo.location || "Not provided"
                }
            </div>
        </div>
    `;

  // Update social links
  const socialLinks = document.getElementById("socialLinks");
  socialLinks.innerHTML = `
        <div class="social-links">
            ${
              currentProfile.socialLinks.linkedin
                ? `<a href="${currentProfile.socialLinks.linkedin}" target="_blank"><i class="bi bi-linkedin"></i> LinkedIn</a>`
                : ""
            }
            ${
              currentProfile.socialLinks.github
                ? `<a href="${currentProfile.socialLinks.github}" target="_blank"><i class="bi bi-github"></i> GitHub</a>`
                : ""
            }
            ${
              currentProfile.socialLinks.portfolio
                ? `<a href="${currentProfile.socialLinks.portfolio}" target="_blank"><i class="bi bi-globe"></i> Portfolio</a>`
                : ""
            }
        </div>
    `;

  // Update education
  const educationList = document.getElementById("educationList");
  educationList.innerHTML =
    currentProfile.education
      .map(
        (edu) => `
    <div class="education-item">
        <h3>${edu.degree || "Degree"}</h3>
        <p>${edu.field || "Field"}</p>
        <p>${edu.startYear || ""} - ${edu.endYear || ""}</p>
        <p>Grade: ${edu.grade || ""}</p>
    </div>
`
      )
      .join("") || "No education details added";

  // Update skills
  const skillsList = document.getElementById("skillsList");
  if (currentProfile.skills && currentProfile.skills.length > 0) {
    skillsList.innerHTML = currentProfile.skills
      .map(
        (category) => `
        <div class="skills-category">
            <h3>${category.category}</h3>
            <div class="skills-items">
                ${category.items
                  .map(
                    (skill) => `
                    <span class="skill-tag">${skill}</span>
                `
                  )
                  .join("")}
            </div>
        </div>
    `
      )
      .join("");
  } else {
    skillsList.innerHTML = "<p>No skills added yet.</p>";
  }

  // experience display
  const experienceList = document.getElementById("experienceList");
  if (currentProfile.experience && currentProfile.experience.length > 0) {
    experienceList.innerHTML = currentProfile.experience
      .map(
        (exp) => `
        <div class="experience-item">
            <h3>${exp.title}</h3>
            <p class="company">${exp.company} - ${exp.location}</p>
            <p class="duration">${formatDate(exp.startDate)} - ${formatDate(
          exp.endDate
        )}</p>
            <p class="description">${exp.description}</p>
            <div class="technologies">
                ${exp.technologies
                  .map(
                    (tech) => `
                    <span class="tech-tag">${tech}</span>
                `
                  )
                  .join("")}
            </div>
        </div>
    `
      )
      .join("");
  } else {
    experienceList.innerHTML = "<p>No experience added yet.</p>";
  }

  // projects display
  const projectsList = document.getElementById("projectsList");
  if (currentProfile.projects && currentProfile.projects.length > 0) {
    projectsList.innerHTML = currentProfile.projects
      .map(
        (project) => `
        <div class="project-item">
            <h3>${project.title}</h3>
            <p class="duration">${formatDate(project.startDate)} - ${formatDate(
          project.endDate
        )}</p>
            <p class="description">${project.description}</p>
            <div class="technologies">
                ${project.technologies
                  .map(
                    (tech) => `
                    <span class="tech-tag">${tech}</span>
                `
                  )
                  .join("")}
            </div>
            ${
              project.link
                ? `
                <div class="project-link">
                    <a href="${project.link}" target="_blank" rel="noopener noreferrer">
                        View Project <i class="fas fa-external-link-alt"></i>
                    </a>
                </div>
            `
                : ""
            }
        </div>
    `
      )
      .join("");
  } else {
    projectsList.innerHTML = "<p>No projects added yet.</p>";
  }

  // extracurriculars display
  const extracurricularsList = document.getElementById("extracurricularsList");
  if (
    currentProfile.extracurriculars &&
    currentProfile.extracurriculars.length > 0
  ) {
    extracurricularsList.innerHTML = currentProfile.extracurriculars
      .map(
        (activity) => `
        <div class="activity-item">
            <h3>${activity.activity}</h3>
            <p class="duration">${formatDate(
              activity.startDate
            )} - ${formatDate(activity.endDate)}</p>
            <p class="description">${activity.description}</p>
        </div>
    `
      )
      .join("");
  } else {
    extracurricularsList.innerHTML =
      "<p>No extracurricular activities added yet.</p>";
  }
}
// Initialize tabs
function initializeTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const tabPanes = document.querySelectorAll(".tab-pane");
      tabPanes.forEach((pane) => pane.classList.remove("active"));
      document.getElementById(tab.dataset.tab).classList.add("active");

      activeTab = tab.dataset.tab;
    });
  });
}

// Initialize image upload
function initializeImageUpload() {
  const imageUpload = document.getElementById("imageUpload");
  imageUpload.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    showLoading();
    try {
      const response = await fetch("/api/upload-image", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.filename) {
        document.getElementById(
          "profileImage"
        ).src = `/static/uploads/profile_images/${data.filename}`;
      }
    } catch (error) {
      console.error("Error uploading image:", error);
      showError("Failed to upload image");
    }
    hideLoading();
  });
}

// Initialize form handlers
function initializeFormHandlers() {
  const editForm = document.getElementById("editForm");

  editForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    if (isSubmitting) {
      console.log("Form submission already in progress");
      return;
    }

    const formData = new FormData(this);
    const sectionSingular = document
      .getElementById("modalTitle")
      .textContent.toLowerCase()
      .split(" ")[1];

    showLoading();
    try {
      if (sectionSingular === "education") {
        const result = await handleEducationSubmit(formData);
        if (result.success) {
          await loadProfile();
          document.getElementById("editModal").style.display = "none";
        }
      } else {
        // Handle other sections...
      }
    } catch (error) {
      console.error("Error:", error);
      showAlert(error.message || "Failed to update profile", "error");
    } finally {
      hideLoading();
    }
  });
}

// Modal functions
function initializeModals() {
  const modal = document.getElementById("editModal");
  const closeBtn = document.querySelector(".close");

  closeBtn.onclick = () => (modal.style.display = "none");
  window.onclick = (e) => {
    if (e.target === modal) modal.style.display = "none";
  };
}

// Generate form fields based on section
function generateFormFields(section) {
  switch (section) {
    case "personal":
      return `
                <div class="form-group">
                    <label>Name</label>
                    <input type="text" name="name" value="${
                      currentProfile.personalInfo.name
                    }">
                </div>
                <div class="form-group">
                    <label>Bio</label>
                    <textarea name="bio">${
                      currentProfile.personalInfo.bio || ""
                    }</textarea>
                </div>
                <div class="form-group">
                    <label>Phone</label>
                    <input type="tel" name="phone" value="${
                      currentProfile.personalInfo.phone || ""
                    }">
                </div>
                <div class="form-group">
                    <label>Location</label>
                    <input type="text" name="location" value="${
                      currentProfile.personalInfo.location || ""
                    }">
                </div>
                <button type="submit" class="submit-btn">Save Changes</button>
            `;

    case "social":
      return `
                <div class="form-group">
                    <label>LinkedIn</label>
                    <input type="url" name="linkedin" value="${
                      currentProfile.socialLinks.linkedin || ""
                    }">
                </div>
                <div class="form-group">
                    <label>GitHub</label>
                    <input type="url" name="github" value="${
                      currentProfile.socialLinks.github || ""
                    }">
                </div>
                <div class="form-group">
                    <label>Portfolio</label>
                    <input type="url" name="portfolio" value="${
                      currentProfile.socialLinks.portfolio || ""
                    }">
                </div>
                <button type="submit" class="submit-btn">Save Changes</button>
            `;

    case "education":
      return `
                <div class="form-group">
                    <label>Institution</label>
                    <input type="text" name="institution" required>
                </div>
                <div class="form-group">
                    <label>Degree</label>
                    <input type="text" name="degree" required>
                </div>
                <div class="form-group">
                    <label>Field</label>
                    <input type="text" name="field" required>
                </div>
                <div class="form-group">
                    <label>Start Year</label>
                    <input type="number" name="startYear" required min="1900" max="2099">
                </div>
                <div class="form-group">
                    <label>End Year</label>
                    <input type="number" name="endYear" required min="1900" max="2099">
                </div>
                <div class="form-group">
                    <label>Grade</label>
                    <input type="text" name="grade" required>
                </div>
                <button type="submit" class="submit-btn">Add Education</button>
            `;

    case "skills":
      return `
                <div class="form-group">
                    <label>Skill Name</label>
                    <input type="text" name="name" required>
                </div>
                <div class="form-group">
                    <label>Proficiency (%)</label>
                    <input type="number" name="proficiency" required min="0" max="100">
                </div>
                <button type="submit" class="submit-btn">Add Skill</button>
            `;

    case "experience":
      return `
                <div class="form-group">
                    <label>Company</label>
                    <input type="text" name="company" required>
                </div>
                <div class="form-group">
                    <label>Position</label>
                    <input type="text" name="position" required>
                </div>
                <div class="form-group">
                    <label>Start Date</label>
                    <input type="date" name="startDate" required>
                </div>
                <div class="form-group">
                    <label>End Date</label>
                    <input type="date" name="endDate" required>
                </div>
                <div class="form-group">
                    <label>Description</label>
                    <textarea name="description" required></textarea>
                </div>
                <button type="submit" class="submit-btn">Add Experience</button>
            `;

    case "projects":
      return `
                <div class="form-group">
                    <label>Project Title</label>
                    <input type="text" name="title" required>
                </div>
                <div class="form-group">
                    <label>Description</label>
                    <textarea name="description" required></textarea>
                </div>
                <div class="form-group">
                    <label>Technologies</label>
                    <input type="text" name="technologies" required>
                </div>
                <div class="form-group">
                    <label>Project Link</label>
                    <input type="url" name="link">
                </div>
                <button type="submit" class="submit-btn">Add Project</button>
            `;
    case "extracurriculars":
      return `
                    <div class="form-group">
                        <label>Title</label>
                        <input type="text" name="title" required>
                    </div>
                    <div class="form-group">
                        <label>Type</label>
                        <select name="type" required>
                            <option value="blog">Blog</option>
                            <option value="certification">Certification</option>
                            <option value="achievement">Achievement</option>
                            <option value="volunteer">Volunteer Work</option>
                            <option value="other">Other</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Description</label>
                        <textarea name="description" required></textarea>
                    </div>
                    <div class="form-group">
                        <label>Date</label>
                        <input type="date" name="date" required>
                    </div>
                    <div class="form-group">
                        <label>Link (Optional)</label>
                        <input type="url" name="link" placeholder="https://...">
                    </div>
                    <button type="submit" class="submit-btn">Add Extracurricular</button>
                `;
  }
}

// Function to load education data
function loadEducation() {
  fetch("/api/profile")
    .then((response) => response.json())
    .then((data) => {
      const educationList = document.getElementById("educationList");
      if (data.education && data.education.length > 0) {
        const educationHTML = data.education
          .map(
            (edu) => `
                    <div class="education-item">
                        <h3>${edu.degree}</h3>
                        <p>${edu.field}</p>
                        <p>${formatDate(edu.startDate)} - ${formatDate(
              edu.endDate
            )}</p>
                        <p>Grade: ${edu.grade}</p>
                    </div>
                `
          )
          .join("");
        educationList.innerHTML = educationHTML;
      } else {
        educationList.innerHTML = "<p>No education details added yet.</p>";
      }
    })
    .catch((error) => {
      console.error("Error loading education:", error);
      document.getElementById("educationList").innerHTML =
        "<p>Failed to load education details.</p>";
    });
}

// Function to format date
function formatDate(dateString) {
  if (!dateString) return "";
  return new Date(dateString).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
  });
}

// Function to handle education form submission
function handleEducationSubmit(formData) {
  const educationData = {
    degree: formData.get("degree"),
    field: formData.get("field"),
    startDate: formData.get("startYear"),
    endDate: formData.get("endYear"),
    grade: formData.get("grade"),
  };

  fetch("/api/profile/education", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify([educationData]), // Send as array
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to update education");
      }
      return response.json();
    })
    .then((result) => {
      if (result.success) {
        closeEditModal();
        loadEducation(); // Reload education section
        showAlert("Education updated successfully!", "success");
      } else {
        throw new Error(result.error || "Failed to update education");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      showAlert("Failed to update education", "error");
    });
}

function openEditModal(section) {
  const modal = document.getElementById("editModal");
  const modalTitle = document.getElementById("modalTitle");
  const editForm = document.getElementById("editForm");

  modal.style.display = "block";
  modalTitle.textContent = `Add ${
    section.charAt(0).toUpperCase() + section.slice(1)
  }`;

  switch (section) {
    case "personal":
      modalTitle.textContent = "Edit Personal Information";
      editForm.innerHTML = `
                <div class="form-group">
                    <label>Name</label>
                    <input type="text" name="name" value="${
                      currentProfile.personalInfo.name
                    }">
                </div>
                <div class="form-group">
                    <label>Bio</label>
                    <textarea name="bio">${
                      currentProfile.personalInfo.bio || ""
                    }</textarea>
                </div>
                <div class="form-group">
                    <label>Phone</label>
                    <input type="tel" name="phone" value="${
                      currentProfile.personalInfo.phone || ""
                    }">
                </div>
                <div class="form-group">
                    <label>Location</label>
                    <input type="text" name="location" value="${
                      currentProfile.personalInfo.location || ""
                    }">
                </div>
                <div class="form-actions">
                    <button type="submit" class="submit-btn">Save Changes</button>
                </div>
            `;
      break;

    case "social":
      modalTitle.textContent = "Edit Social Links";
      editForm.innerHTML = `
                <div class="form-group">
                    <label>LinkedIn</label>
                    <input type="url" name="linkedin" value="${
                      currentProfile.socialLinks.linkedin || ""
                    }">
                </div>
                <div class="form-group">
                    <label>GitHub</label>
                    <input type="url" name="github" value="${
                      currentProfile.socialLinks.github || ""
                    }">
                </div>
                <div class="form-group">
                    <label>Portfolio</label>
                    <input type="url" name="portfolio" value="${
                      currentProfile.socialLinks.portfolio || ""
                    }">
                </div>
                <div class="form-actions">
                    <button type="submit" class="submit-btn">Save Changes</button>
                </div>
            `;
      break;

    case "education":
      modalTitle.textContent = "Add Education";
      editForm.innerHTML = `
                <div class="form-group">
                    <label for="degree">Degree</label>
                    <input type="text" id="degree" name="degree" required>
                </div>
                <div class="form-group">
                    <label for="field">Field of Study</label>
                    <input type="text" id="field" name="field" required>
                </div>
                <div class="form-group">
                    <label for="startYear">Start Year</label>
                    <input type="month" id="startYear" name="startYear" required placeholder="e.g., 2020">
                </div>
                <div class="form-group">
                    <label for="endYear">End Year</label>
                    <input type="month" id="endYear" name="endYear" required placeholder="e.g., 2020">
                </div>
                <div class="form-group">
                    <label for="grade">Grade/CGPA</label>
                    <input type="text" id="grade" name="grade" required>
                </div>
                <div class="form-actions">
                    <button type="submit" class="submit-btn">Save Changes</button>
                    <button type="button" class="cancel-btn" onclick="closeEditModal()">Cancel</button>
                </div>
            `;
      break;
    case "skills":
      modalTitle.textContent = "Add Skills";
      editForm.innerHTML = `
                    <div class="skill-category-container">
                        <div class="form-group">
                            <label for="programmingCategory">Programming Languages</label>
                            <input type="text" id="programmingSkills" name="programming" 
                                   placeholder="Enter skills separated by commas (e.g., Python, JavaScript, Java)">
                        </div>
                        <div class="form-group">
                            <label for="webCategory">Web Technologies</label>
                            <input type="text" id="webSkills" name="web" 
                                   placeholder="Enter skills separated by commas (e.g., HTML, CSS, React)">
                        </div>
                    </div>
                    <div class="form-actions">
                        <button type="submit" class="submit-btn">Save Changes</button>
                        <button type="button" class="cancel-btn" onclick="closeEditModal()">Cancel</button>
                    </div>
                `;
      break;
    case "experience":
      modalTitle.textContent = "Add Experience";
      editForm.innerHTML = `
                    <div class="form-group">
                        <label for="title">Job Title</label>
                        <input type="text" id="title" name="title" required>
                    </div>
                    <div class="form-group">
                        <label for="company">Company</label>
                        <input type="text" id="company" name="company" required>
                    </div>
                    <div class="form-group">
                        <label for="location">Location</label>
                        <input type="text" id="location" name="location" required>
                    </div>
                    <div class="form-group">
                        <label for="startDate">Start Date</label>
                        <input type="month" id="startDate" name="startDate" required>
                    </div>
                    <div class="form-group">
                        <label for="endDate">End Date</label>
                        <input type="month" id="endDate" name="endDate" required>
                    </div>
                    <div class="form-group">
                        <label for="description">Description</label>
                        <textarea id="description" name="description" required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="technologies">Technologies Used</label>
                        <input type="text" id="technologies" name="technologies" 
                               placeholder="Enter technologies separated by commas (e.g., Python, Flask, MongoDB)" required>
                    </div>
                    <div class="form-actions">
                        <button type="submit" class="submit-btn">Save Changes</button>
                        <button type="button" class="cancel-btn" onclick="closeEditModal()">Cancel</button>
                    </div>
                `;
      break;
    case "projects":
      modalTitle.textContent = "Add Project";
      editForm.innerHTML = `
                    <div class="form-group">
                        <label for="projectTitle">Project Title</label>
                        <input type="text" id="projectTitle" name="title" required>
                    </div>
                    <div class="form-group">
                        <label for="projectDescription">Description</label>
                        <textarea id="projectDescription" name="description" required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="projectTechnologies">Technologies Used</label>
                        <input type="text" id="projectTechnologies" name="technologies" 
                               placeholder="Enter technologies separated by commas" required>
                    </div>
                    <div class="form-group">
                        <label for="projectLink">Project Link</label>
                        <input type="url" id="projectLink" name="link" 
                               placeholder="e.g., https://github.com/username/project">
                    </div>
                    <div class="form-group">
                        <label for="projectStartDate">Start Date</label>
                        <input type="month" id="projectStartDate" name="startDate" required>
                    </div>
                    <div class="form-group">
                        <label for="projectEndDate">End Date</label>
                        <input type="month" id="projectEndDate" name="endDate" required>
                    </div>
                    <div class="form-actions">
                        <button type="submit" class="submit-btn">Save Changes</button>
                        <button type="button" class="cancel-btn" onclick="closeEditModal()">Cancel</button>
                    </div>
                `;
      break;
    case "extracurriculars":
      modalTitle.textContent = "Add Extracurricular Activity";
      editForm.innerHTML = `
                    <div class="form-group">
                        <label for="activity">Activity Name</label>
                        <input type="text" id="activity" name="activity" required>
                    </div>
                    <div class="form-group">
                        <label for="description">Description</label>
                        <textarea id="description" name="description" required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="startDate">Start Date</label>
                        <input type="month" id="startDate" name="startDate" required>
                    </div>
                    <div class="form-group">
                        <label for="endDate">End Date</label>
                        <input type="month" id="endDate" name="endDate" required>
                    </div>
                    <div class="form-actions">
                        <button type="submit" class="submit-btn">Save Changes</button>
                        <button type="button" class="cancel-btn" onclick="closeEditModal()">Cancel</button>
                    </div>
                `;
      break;
  }

  // Set up form submission handler
  editForm.onsubmit = async function (e) {
    e.preventDefault();
    const formData = new FormData(e.target);

    try {
      switch (section) {
        case "personal":
          await handlePersonalSubmit(formData);
          break;
        case "social":
          await handleSocialSubmit(formData);
          break;
        case "education":
          await handleEducationSubmit(formData);
          break;
        case "skills":
          await handleSkillsSubmit(formData);
          break;
        case "experience":
          await handleExperienceSubmit(formData);
          break;
        case "projects": // Make sure this matches exactly
          await handleProjectsSubmit(formData);
          break;
        case "extracurriculars":
          await handleExtracurricularsSubmit(formData);
          break;

        default:
          throw new Error("Unknown section");
      }
    } catch (error) {
      console.error("Error:", error);
      showAlert("Failed to update " + section, "error");
    }
  };
}

// Add this to your initialization code
document.addEventListener("DOMContentLoaded", function () {
  loadEducation();
  // ... other initialization code ...
});

// Utility functions
function showLoading() {
  document.getElementById("loadingOverlay").classList.add("active");
}

function hideLoading() {
  document.getElementById("loadingOverlay").classList.remove("active");
}

function showError(message) {
  alert(message); // You can replace this with a better error notification system
}

// Add these submit handler functions after your openEditModal function

async function handlePersonalSubmit(formData) {
  try {
    const response = await fetch("/api/profile/personal", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(Object.fromEntries(formData)),
    });

    if (!response.ok) {
      throw new Error("Failed to update personal information");
    }

    const result = await response.json();
    if (result.success) {
      closeEditModal();
      await loadProfile();
      showAlert("Personal information updated successfully!", "success");
    }
  } catch (error) {
    console.error("Error:", error);
    showAlert("Failed to update personal information", "error");
  }
}

async function handleSocialSubmit(formData) {
  try {
    const response = await fetch("/api/profile/social", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(Object.fromEntries(formData)),
    });

    if (!response.ok) {
      throw new Error("Failed to update social links");
    }

    const result = await response.json();
    if (result.success) {
      closeEditModal();
      await loadProfile();
      showAlert("Social links updated successfully!", "success");
    }
  } catch (error) {
    console.error("Error:", error);
    showAlert("Failed to update social links", "error");
  }
}

let isSubmitting = false; // Add this flag at the top of your file

async function handleEducationSubmit(formData) {
  // Prevent double submission
  if (isSubmitting) {
    console.log("Submission already in progress");
    return;
  }

  isSubmitting = true;

  try {
    const educationData = {
      degree: formData.get("degree"),
      field: formData.get("field"),
      startYear: formData.get("startYear"),
      endYear: formData.get("endYear"),
      grade: formData.get("grade"),
      institution: formData.get("institution") || "",
      activities: formData.get("activities") || "",
    };

    console.log("Submitting education data:", educationData);

    const response = await fetch("/api/profile/education", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(educationData),
      credentials: "include",
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || "Failed to update education");
    }

    return data;
  } catch (error) {
    console.error("Error in handleEducationSubmit:", error);
    throw error;
  } finally {
    isSubmitting = false; // Reset the flag regardless of success/failure
  }
}

async function handleExtracurricularsSubmit(formData) {
  try {
    const extracurricularData = {
      activity: formData.get("activity"),
      description: formData.get("description"),
      startDate: formData.get("startDate"),
      endDate: formData.get("endDate"),
    };

    console.log("Sending extracurricular data:", extracurricularData); // Debug log

    const response = await fetch("/api/profile/extracurriculars", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify([extracurricularData]), // Send as array
    });

    console.log("Response status:", response.status); // Debug log

    if (!response.ok) {
      throw new Error("Failed to update extracurriculars");
    }

    const result = await response.json();
    console.log("Response result:", result); // Debug log

    if (result.success) {
      closeEditModal();
      await loadProfile();
      showAlert("Extracurricular activity added successfully!", "success");
      setTimeout(() => loadProfile(), 100);
    } else {
      throw new Error(result.error || "Failed to update extracurriculars");
    }
  } catch (error) {
    console.error("Error:", error);
    showAlert("Failed to add extracurricular activity", "error");
  }
}

async function handleSkillsSubmit(formData) {
  // Format skills data to match your MongoDB structure
  const skillsData = [
    {
      category: "Programming Languages",
      items: formData
        .get("programming")
        .split(",")
        .map((skill) => skill.trim())
        .filter((skill) => skill !== ""),
    },
    {
      category: "Web Technologies",
      items: formData
        .get("web")
        .split(",")
        .map((skill) => skill.trim())
        .filter((skill) => skill !== ""),
    },
  ];

  try {
    const response = await fetch("/api/profile/skills", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(skillsData),
    });

    if (!response.ok) {
      throw new Error("Failed to update skills");
    }

    const result = await response.json();
    if (result.success) {
      closeEditModal();
      await loadProfile(); // Refresh the profile data
      showAlert("Skills updated successfully!", "success");
    }
  } catch (error) {
    console.error("Error:", error);
    showAlert("Failed to update skills", "error");
  }
}

async function handleProjectsSubmit(formData) {
  try {
    const projectData = {
      title: formData.get("title"),
      description: formData.get("description"),
      technologies: formData
        .get("technologies")
        .split(",")
        .map((tech) => tech.trim())
        .filter((tech) => tech !== ""),
      link: formData.get("link"),
      startDate: formData.get("startDate"),
      endDate: formData.get("endDate"),
    };

    console.log("Sending project data:", projectData); // Debug log
    console.log("Making request to:", "/api/profile/projects");
    let response = await fetch("/api/profile/projects", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify([projectData]),
    });

    if (response.ok) {
      closeEditModal();
      showAlert("Project added successfully!", "success");
      // Add delay before reload
      await new Promise((resolve) => setTimeout(resolve, 500));
      await loadProfile();
      return;
    }
    throw new Error("Failed to update project");
  } catch (error) {
    console.error("Error:", error);
    showAlert("Failed to add project", "error");
  }
}

async function handleExperienceSubmit(formData) {
  const experienceData = {
    title: formData.get("title"),
    company: formData.get("company"),
    location: formData.get("location"),
    startDate: formData.get("startDate"),
    endDate: formData.get("endDate"),
    description: formData.get("description"),
    technologies: formData
      .get("technologies")
      .split(",")
      .map((tech) => tech.trim())
      .filter((tech) => tech !== ""),
  };

  try {
    const response = await fetch("/api/profile/experience", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify([experienceData]),
    });

    if (!response.ok) {
      throw new Error("Failed to update experience");
    }

    const result = await response.json();
    if (result.success) {
      closeEditModal();
      await loadProfile();
      showAlert("Experience updated successfully!", "success");
    }
  } catch (error) {
    console.error("Error:", error);
    showAlert("Failed to update experience", "error");
  }
}

// Add this utility function if you don't have it
function showAlert(message, type = "error") {
  const alertDiv = document.createElement("div");
  alertDiv.className = `alert alert-${type}`;
  alertDiv.textContent = message;

  // Remove any existing alerts
  const existingAlert = document.querySelector(".alert");
  if (existingAlert) {
    existingAlert.remove();
  }

  // Add the new alert
  document.body.insertBefore(alertDiv, document.body.firstChild);

  // Remove the alert after 3 seconds
  setTimeout(() => {
    alertDiv.remove();
  }, 3000);
}

// Add this if you don't have it
function closeEditModal() {
  document.getElementById("editModal").style.display = "none";
}
