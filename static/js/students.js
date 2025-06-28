document.addEventListener("DOMContentLoaded", function () {
    const searchForm = document.getElementById("searchForm");
    const searchResults = document.getElementById("searchResults");
    const studentModal = new bootstrap.Modal(document.getElementById('studentModal'));

    searchForm.addEventListener("submit", async function (e) {
        e.preventDefault();
        console.log('Form submitted');
        const query = document.getElementById("searchQuery").value;
        console.log("Search query:", query);  // Debug statement

        if (query.trim() === "") {
            alert("Please enter a search term.");
            return;
        }

        try {
            const response = await fetch("/api/search", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ query: query }),
            });
            console.log("API response:", response);  // Debug statement

            if (!response.ok) {
                throw new Error("Search failed.");
            }

            const data = await response.json();
            console.log("Search results:", data);  // Debug statement
            displayResults(data);
        } catch (error) {
            console.error("Error:", error);
            searchResults.innerHTML = '<p class="text-center text-danger">An error occurred while searching.</p>';
        }
    });

    function displayResults(results) {
        searchResults.innerHTML = "";

        if (!Array.isArray(results) || results.length === 0) {
            searchResults.innerHTML = "<p class='text-center'>No results found.</p>";
            return;
        }

        results.forEach((result) => {
            const card = document.createElement("div");
            card.className = "col-md-4 mb-4";
            card.innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">${result.personalInfo?.name || 'No Name'}</h5>
                        <p class="card-text">
                            <strong>Skills:</strong> ${result.skills?.join(', ') || 'No skills listed'}
                        </p>
                        <p class="card-text">
                            <strong>Interests:</strong> ${result.personalInfo?.interests?.join(', ') || 'No interests listed'}
                        </p>
                        <button class="btn btn-primary btn-sm" onclick="showStudentDetails('${result._id}')">
                            View Details
                        </button>
                    </div>
                </div>
            `;
            searchResults.appendChild(card);
        });
    }

         // Event delegation for view details buttons
    searchResults.addEventListener('click', async function (e) {
        if (e.target.classList.contains('view-details')) {
            const studentId = e.target.dataset.studentId;
            try {
                const response = await fetch(`/api/student/${studentId}`);
                if (!response.ok) throw new Error('Failed to fetch student details');

                const student = await response.json();
                const modalBody = document.getElementById('studentDetails');

                modalBody.innerHTML = `
                    <div class="student-details">
                        <div class="row">
                            <div class="col-md-4">
                                <img src="${student.personalInfo?.profileImage || '/static/img/default.jpg'}" 
                                     class="img-fluid rounded" 
                                     alt="${student.personalInfo?.name || 'Student'}'s profile picture">
                            </div>
                            <div class="col-md-8">
                                <h4>${student.personalInfo?.name || 'No Name'}</h4>
                                <p><strong>Email:</strong> ${student.personalInfo?.email || 'Not provided'}</p>
                                <p><strong>Skills:</strong> ${student.skills?.join(', ') || 'No skills listed'}</p>
                                <p><strong>Education:</strong> ${student.education?.map(edu => 
                                    `${edu.degree} in ${edu.field}`).join(', ') || 'No education listed'}</p>
                                <p><strong>Projects:</strong> ${student.projects?.map(proj => 
                                    proj.name).join(', ') || 'No projects listed'}</p>
                            </div>
                        </div>
                    </div>
                `;

                const modal = new bootstrap.Modal(document.getElementById('studentModal'));
                modal.show();
            } catch (error) {
                console.error('Error fetching student details:', error);
                alert('Failed to load student details');
            }
        }
    });

    window.showStudentDetails = async function (studentId) {
        try {
            const response = await fetch(`/api/student/${studentId}`);
            if (!response.ok) {
                throw new Error("Failed to fetch student details.");
            }

            const student = await response.json();
            const modalBody = document.getElementById("studentDetails");
            modalBody.innerHTML = `
                <p><strong>Name:</strong> ${student.name}</p>
                <p><strong>Designation:</strong> ${student.designation}</p>
                <p><strong>Degree:</strong> ${student.degree}</p>
                <p><strong>Department:</strong> ${student.department}</p>
                <p><strong>Skills:</strong> ${student.skills.join(", ")}</p>
                <p><strong>Projects:</strong> ${student.projects.join(", ")}</p>
                <p><strong>Certifications:</strong> ${student.certifications.join(", ")}</p>
            `;

            const modal = new bootstrap.Modal(document.getElementById("studentModal"));
            modal.show();
        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred while fetching student details.");
        }
    };
});