document.addEventListener("DOMContentLoaded", function () {
    const chatMessages = document.getElementById("chat-messages");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");

    // Focus on input when page loads
    userInput.focus();

    // Function to add a message to the chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);

        // Process markdown-like code blocks
        let processedText = text;

        // Handle code blocks with ```
        processedText = processedText.replace(
            /```([a-z]*)([\s\S]*?)```/g,
            function (match, language, code) {
                return `<pre><code class="${language}">${code.trim()}</code></pre>`;
            }
        );

        // Handle inline code with `
        processedText = processedText.replace(/`([^`]+)`/g, "<code>$1</code>");

        messageDiv.innerHTML = processedText;
        chatMessages.appendChild(messageDiv);

        // Scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to show loading indicator
    function showLoading() {
        const loadingDiv = document.createElement("div");
        loadingDiv.classList.add("message", "bot", "loading");
        loadingDiv.id = "loading-indicator";

        const typingIndicator = document.createElement("div");
        typingIndicator.classList.add("typing-indicator");

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement("span");
            typingIndicator.appendChild(dot);
        }

        loadingDiv.appendChild(typingIndicator);
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to hide loading indicator
    function hideLoading() {
        const loadingIndicator = document.getElementById("loading-indicator");
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }

    // Function to send message
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessage(message, "user");

        // Clear input
        userInput.value = "";

        // Show loading indicator
        showLoading();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message }),
            });

            const data = await response.json();

            // Hide loading indicator
            hideLoading();

            if (data.error) {
                console.error("Error:", data.error);
                addMessage(
                    "I'm sorry, I encountered an error. Please try again.",
                    "bot"
                );
            } else {
                addMessage(data.response, "bot");
            }
        } catch (error) {
            console.error("Error:", error);
            hideLoading();
            addMessage(
                "I'm sorry, I couldn't connect to the server. Please try again.",
                "bot"
            );
        }
    }

    // Event listeners
    sendButton.addEventListener("click", sendMessage);

    userInput.addEventListener("keypress", function (e) {
        if (e.key === "Enter") {
            sendMessage();
        }
    });
});
