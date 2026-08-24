/**
 * Aster & Row Customer Support Agent Frontend Application
 */

(function () {
    "use strict";

    const API_ENDPOINT = "/api/chat";
    const SESSION_STORAGE_KEY = "aster_row_session_id";

    // DOM Elements
    const chatMessages = document.getElementById("chat-messages");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const newChatBtn = document.getElementById("new-chat-btn");
    const sessionBadge = document.getElementById("session-badge");

    // Debug Panel Elements
    const handoffIndicator = document.getElementById("handoff-indicator");
    const sourcesList = document.getElementById("sources-list");
    const sourcesCount = document.getElementById("sources-count");
    const toolCallsList = document.getElementById("tool-calls-list");
    const toolCallsCount = document.getElementById("tool-calls-count");

    let currentSessionId = "";
    let isSubmitting = false;

    /**
     * Initialize or restore session ID from sessionStorage.
     */
    function initSession() {
        let sid = sessionStorage.getItem(SESSION_STORAGE_KEY);
        if (!sid || sid.trim() === "") {
            sid = generateUUID();
            sessionStorage.setItem(SESSION_STORAGE_KEY, sid);
        }
        currentSessionId = sid;
        updateSessionDisplay();
    }

    /**
     * Generate a unique session UUID.
     */
    function generateUUID() {
        if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
            return crypto.randomUUID();
        }
        return "sess-" + Math.random().toString(36).substring(2, 10) + "-" + Date.now();
    }

    /**
     * Update session badge UI.
     */
    function updateSessionDisplay() {
        if (sessionBadge) {
            const shortId = currentSessionId.length > 12 
                ? currentSessionId.substring(0, 10) + "..." 
                : currentSessionId;
            sessionBadge.textContent = "Session: " + shortId;
            sessionBadge.title = "Active Session ID: " + currentSessionId;
        }
    }

    /**
     * Start a fresh conversation session.
     */
    function startNewConversation() {
        currentSessionId = generateUUID();
        sessionStorage.setItem(SESSION_STORAGE_KEY, currentSessionId);
        updateSessionDisplay();

        // Clear chat messages to default welcome message
        chatMessages.innerHTML = `
            <div class="message assistant-message welcome-message">
                <div class="message-avatar" aria-hidden="true">🌲</div>
                <div class="message-body">
                    <div class="message-sender">Aster &amp; Row Support</div>
                    <div class="message-content">
                        <p>Hello! I am your Aster &amp; Row customer support assistant. I can help look up order tracking, explain return policies, warranty terms, and provide product care instructions.</p>
                        <p class="suggestion-prompt"><strong>Quick examples you can ask:</strong></p>
                        <ul class="suggestion-list">
                            <li><button class="suggestion-chip" data-msg="Where is ORD-1007?">"Where is ORD-1007?"</button></li>
                            <li><button class="suggestion-chip" data-msg="What are the warranty periods for bags and drinkware?">"What are the warranty periods for bags and drinkware?"</button></li>
                            <li><button class="suggestion-chip" data-msg="How long do I have to return an unused backpack?">"How long do I have to return an unused backpack?"</button></li>
                        </ul>
                    </div>
                </div>
            </div>
        `;

        resetDebugPanel();
        userInput.value = "";
        userInput.focus();
    }

    /**
     * Reset turn inspection debug panel to initial empty state.
     */
    function resetDebugPanel() {
        if (handoffIndicator) {
            handoffIndicator.className = "handoff-status handoff-none";
            handoffIndicator.innerHTML = `
                <span class="handoff-icon" aria-hidden="true">✓</span>
                <span class="handoff-text">No Handoff Required</span>
            `;
        }
        if (sourcesList && sourcesCount) {
            sourcesCount.textContent = "0";
            sourcesList.innerHTML = `<div class="empty-state">No policy sources cited in latest turn.</div>`;
        }
        if (toolCallsList && toolCallsCount) {
            toolCallsCount.textContent = "0";
            toolCallsList.innerHTML = `<div class="empty-state">No tool calls executed in latest turn.</div>`;
        }
    }

    /**
     * Safely escape HTML characters.
     */
    function escapeHTML(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    /**
     * Safe basic Markdown parser for bold, italic, lists, and inline citations.
     */
    function formatSafeMarkdown(text) {
        if (!text) return "";

        // First escape HTML entities
        let safe = escapeHTML(text);

        // Bold: **text**
        safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

        // Italic: *text* or _text_
        safe = safe.replace(/\*([^\*]+)\*/g, "<em>$1</em>");

        // Format citation brackets: [filename — section]
        safe = safe.replace(/\[([a-zA-Z0-9_\-\.\s—]+)\]/g, '<span class="source-badge-inline">[$1]</span>');

        // Split paragraphs by double newline
        const paragraphs = safe.split(/\n\s*\n/);
        const formattedBlocks = paragraphs.map(p => {
            const lines = p.trim().split("\n");
            const isList = lines.every(l => l.trim().startsWith("- ") || l.trim().startsWith("* ") || l.trim().startsWith("• "));

            if (isList) {
                const listItems = lines.map(l => `<li>${l.replace(/^[-*•]\s+/, "")}</li>`).join("");
                return `<ul>${listItems}</ul>`;
            } else {
                return `<p>${lines.join("<br>")}</p>`;
            }
        });

        return formattedBlocks.join("");
    }

    /**
     * Append a message bubble to the chat container.
     */
    function appendMessage(sender, text, isUser = false) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${isUser ? "user-message" : "assistant-message"}`;

        const avatarDiv = document.createElement("div");
        avatarDiv.className = "message-avatar";
        avatarDiv.setAttribute("aria-hidden", "true");
        avatarDiv.textContent = isUser ? "👤" : "🌲";

        const bodyDiv = document.createElement("div");
        bodyDiv.className = "message-body";

        const senderDiv = document.createElement("div");
        senderDiv.className = "message-sender";
        senderDiv.textContent = sender;

        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";

        if (isUser) {
            // User message is rendered safely with plain text
            const p = document.createElement("p");
            p.textContent = text;
            contentDiv.appendChild(p);
        } else {
            // Assistant message is formatted safely
            contentDiv.innerHTML = formatSafeMarkdown(text);
        }

        bodyDiv.appendChild(senderDiv);
        bodyDiv.appendChild(contentDiv);
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(bodyDiv);

        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    /**
     * Display the animated thinking bubble.
     */
    function showThinkingIndicator() {
        const thinkingDiv = document.createElement("div");
        thinkingDiv.id = "thinking-indicator";
        thinkingDiv.className = "message assistant-message";
        thinkingDiv.innerHTML = `
            <div class="message-avatar" aria-hidden="true">🌲</div>
            <div class="message-body">
                <div class="message-sender">Aster &amp; Row Support</div>
                <div class="thinking-bubble" role="status" aria-label="Thinking">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(thinkingDiv);
        scrollToBottom();
    }

    /**
     * Remove the thinking bubble.
     */
    function removeThinkingIndicator() {
        const indicator = document.getElementById("thinking-indicator");
        if (indicator) {
            indicator.remove();
        }
    }

    /**
     * Smooth scroll chat to bottom.
     */
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    /**
     * Update the debug panel with turn results.
     */
    function updateDebugPanel(sources, toolCalls, handoff) {
        // 1. Update Handoff Status
        if (handoffIndicator) {
            if (handoff) {
                handoffIndicator.className = "handoff-status handoff-required";
                handoffIndicator.innerHTML = `
                    <span class="handoff-icon" aria-hidden="true">⚠️</span>
                    <span class="handoff-text">Human Support Recommended</span>
                `;
            } else {
                handoffIndicator.className = "handoff-status handoff-none";
                handoffIndicator.innerHTML = `
                    <span class="handoff-icon" aria-hidden="true">✓</span>
                    <span class="handoff-text">Self-Service / No Handoff</span>
                `;
            }
        }

        // 2. Update Sources List
        if (sourcesList && sourcesCount) {
            const list = Array.isArray(sources) ? sources : [];
            sourcesCount.textContent = list.length.toString();

            if (list.length === 0) {
                sourcesList.innerHTML = `<div class="empty-state">No policy sources cited in latest turn.</div>`;
            } else {
                sourcesList.innerHTML = list.map(src => {
                    const safeSrc = escapeHTML(src);
                    return `
                        <div class="source-item">
                            <span class="source-icon">📄</span>
                            <span>${safeSrc}</span>
                        </div>
                    `;
                }).join("");
            }
        }

        // 3. Update Tool Calls List
        if (toolCallsList && toolCallsCount) {
            const list = Array.isArray(toolCalls) ? toolCalls : [];
            toolCallsCount.textContent = list.length.toString();

            if (list.length === 0) {
                toolCallsList.innerHTML = `<div class="empty-state">No tool calls executed in latest turn.</div>`;
            } else {
                toolCallsList.innerHTML = list.map(tc => {
                    const toolName = escapeHTML(tc.name || "unknown");
                    const argsJson = escapeHTML(JSON.stringify(tc.arguments || {}, null, 2));
                    return `
                        <div class="tool-call-item">
                            <div class="tool-name-badge">🔧 ${toolName}</div>
                            <pre class="tool-args-pre">${argsJson}</pre>
                        </div>
                    `;
                }).join("");
            }
        }
    }

    /**
     * Submit user message to backend.
     */
    async function sendMessage(text) {
        const message = text.trim();
        if (!message || isSubmitting) return;

        isSubmitting = true;
        userInput.disabled = true;
        sendBtn.disabled = true;

        // Display user message
        appendMessage("You", message, true);
        userInput.value = "";
        userInput.style.height = "auto";

        // Show thinking indicator
        showThinkingIndicator();

        try {
            const response = await fetch(API_ENDPOINT, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    message: message
                })
            });

            removeThinkingIndicator();

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            appendMessage("Aster & Row Support", data.answer || "I apologize, but no answer was returned.");
            updateDebugPanel(data.sources, data.tool_calls, data.handoff);
        } catch (error) {
            removeThinkingIndicator();
            appendMessage(
                "Aster & Row Support",
                "Sorry, I encountered an error while processing your request. Please try again or reach out to Aster & Row support directly."
            );
            updateDebugPanel([], [], true);
        } finally {
            isSubmitting = false;
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    // Event Listeners
    chatForm.addEventListener("submit", function (e) {
        e.preventDefault();
        sendMessage(userInput.value);
    });

    userInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Auto-resize textarea
    userInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 120) + "px";
    });

    newChatBtn.addEventListener("click", function () {
        startNewConversation();
    });

    // Quick suggestion chips handler
    document.addEventListener("click", function (e) {
        if (e.target && e.target.classList.contains("suggestion-chip")) {
            const msg = e.target.getAttribute("data-msg");
            if (msg) {
                sendMessage(msg);
            }
        }
    });

    // Initialize application
    initSession();
})();
