/* RESPONSIVE DU FICHIER BASE*/
document.addEventListener('DOMContentLoaded', function() {
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    let isCollapsed = false;

    function toggleSidebar() {
        isCollapsed = !isCollapsed;
        sidebar.classList.toggle('collapsed', isCollapsed);
        mainContent.classList.toggle('expanded', isCollapsed);
        
        // Sauvegarder l'état
        localStorage.setItem('sidebarState', isCollapsed ? 'collapsed' : 'expanded');
    }

    sidebarToggle.addEventListener('click', toggleSidebar);

    // Restaurer l'état initial
    const savedState = localStorage.getItem('sidebarState');
    if (savedState === 'collapsed') {
        toggleSidebar();
    }
});
document.addEventListener('DOMContentLoaded', function() {
    const contentWrapper = document.querySelector('.content-wrapper');
    let isScrolling = false;
    let scrollTimer;

    // Fonction pour ajouter la classe lors du défilement
    contentWrapper.addEventListener('scroll', function() {
        if (!isScrolling) {
            isScrolling = true;
            contentWrapper.classList.add('show-scrollbar');
        }
        
        // Réinitialiser le timer à chaque événement de défilement
        clearTimeout(scrollTimer);
        
        // Définir un timer pour supprimer la classe après l'arrêt du défilement
        scrollTimer = setTimeout(function() {
            isScrolling = false;
            contentWrapper.classList.remove('show-scrollbar');
        }, 1000); // La barre disparaît 1 seconde après l'arrêt du défilement
    });
});

// Fonctionnalités globales pour l'application Finn

document.addEventListener('DOMContentLoaded', function() {
    // Toggle pour le menu latéral
    const sidebarToggle = document.getElementById('sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            document.querySelector('.app-container').classList.toggle('sidebar-collapsed');
        });
    }

    // Chargement des conversations récentes
    loadRecentConversations();

    // Auto-resize des textareas
    setupTextareaAutoResize();

    // Initialiser la mise en évidence du menu actif
    highlightActiveMenu();
    
    // Initialiser la gestion de la boîte de dialogue de suppression
    setupDeleteConfirmation();
});

/**
 * Charge les conversations récentes via API et les affiche dans la sidebar
 */
function loadRecentConversations() {
    const conversationsContainer = document.getElementById('recentConversations');
    if (!conversationsContainer) return;

    fetch('/api/conversations')
        .then(response => response.json())
        .then(conversations => {
            if (Object.keys(conversations).length === 0) {
                conversationsContainer.innerHTML = '<div class="no-conversations">No recent conversations</div>';
                return;
            }

            let conversationsHtml = '';
            
            // Parcourir les conversations (max 10 pour éviter de surcharger la sidebar)
            Object.entries(conversations).slice(0, 10).forEach(([id, conv]) => {
                conversationsHtml += `
                <div class="nav-item conversation-item" data-conversation-id="${id}">
                    <a href="/chat/${id}" class="nav-link conversation-link">
                        <div class="nav-content">
                            <svg class="nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                            </svg>
                            <span class="conversation-title">${conv.title}</span>
                        </div>
                        <div class="conversation-actions">
                            <button class="delete-conversation" data-conversation-id="${id}">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M3 6h18"></path>
                                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"></path>
                                    <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                                </svg>
                            </button>
                        </div>
                    </a>
                </div>`;
            });

            conversationsContainer.innerHTML = conversationsHtml;

            // Attacher les gestionnaires d'événements pour les boutons de suppression
            document.querySelectorAll('.delete-conversation').forEach(button => {
                button.addEventListener('click', handleDeleteButtonClick);
            });
        })
        .catch(error => {
            console.error('Error loading conversations:', error);
            conversationsContainer.innerHTML = '<div class="no-conversations">Failed to load conversations</div>';
        });
}

/**
 * Configure la boîte de dialogue de confirmation de suppression
 */
function setupDeleteConfirmation() {
    const modal = document.getElementById('deleteConfirmModal');
    const cancelBtn = document.getElementById('cancelDeleteBtn');
    const confirmBtn = document.getElementById('confirmDeleteBtn');
    
    if (!modal || !cancelBtn || !confirmBtn) return;
    
    // Gérer le bouton d'annulation
    cancelBtn.addEventListener('click', function() {
        modal.style.display = 'none';
        window.currentConversationToDelete = null;
    });
    
    // Gérer le bouton de confirmation
    confirmBtn.addEventListener('click', function() {
        if (window.currentConversationToDelete) {
            deleteConversation(window.currentConversationToDelete);
        }
        modal.style.display = 'none';
        window.currentConversationToDelete = null;
    });
    
    // Fermer la modal si on clique en dehors
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
            window.currentConversationToDelete = null;
        }
    });
}

/**
 * Gère le clic sur le bouton de suppression
 */
function handleDeleteButtonClick(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const conversationId = this.dataset.conversationId;
    window.currentConversationToDelete = conversationId;
    
    // Afficher la boîte de dialogue de confirmation
    const modal = document.getElementById('deleteConfirmModal');
    if (modal) {
        modal.style.display = 'flex';
    }
}

/**
 * Supprime une conversation après confirmation
 */
function deleteConversation(conversationId) {
    fetch(`/api/conversations/${conversationId}`, {
        method: 'DELETE',
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'deleted') {
            // Si on est sur la page de cette conversation, rediriger vers la page de chat
            const currentConversation = document.querySelector('.chat-container')?.dataset.conversationId;
            
            if (currentConversation === conversationId) {
                window.location.href = '/chat';
            } else {
                // Sinon, juste supprimer l'élément du DOM
                document.querySelector(`.conversation-item[data-conversation-id="${conversationId}"]`).remove();
                
                // Si plus aucune conversation, afficher le message
                if (document.querySelectorAll('.conversation-item').length === 0) {
                    document.getElementById('recentConversations').innerHTML = 
                        '<div class="no-conversations">No recent conversations</div>';
                }
            }
        }
    })
    .catch(error => console.error('Error deleting conversation:', error));
}

/**
 * Configure l'auto-resize pour tous les textareas de l'application
 */
function setupTextareaAutoResize() {
    document.querySelectorAll('textarea').forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Initialiser la hauteur
        if (textarea.value) {
            textarea.dispatchEvent(new Event('input'));
        }
    });
}

/**
 * Met en évidence l'élément de menu actif
 */
function highlightActiveMenu() {
    const currentPath = window.location.pathname;
    
    // Supprimer toutes les classes active
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Trouver et mettre en évidence le lien actif
    let activeLink;
    
    if (currentPath.startsWith('/chat')) {
        activeLink = document.querySelector('.new-chat');
    } else if (currentPath.startsWith('/notifications')) {
        activeLink = document.querySelector('a[href="/notifications"]');
    } else if (currentPath.startsWith('/news')) {
        activeLink = document.querySelector('a[href="/news"]');
    } else if (currentPath.startsWith('/simulator')) {
        activeLink = document.querySelector('a[href="/simulator"]');
    }
    
    if (activeLink) {
        activeLink.classList.add('active');
    }
}

/**
 * Si on est sur la page de chat, initialiser les gestionnaires spécifiques
 */
if (document.querySelector('.chat-container')) {
    initializeChatFunctions();
}

/**
 * Initialise les fonctionnalités spécifiques au chat
 */
function initializeChatFunctions() {
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const chatMessages = document.getElementById('chatMessages');
    const conversationContainer = document.querySelector('.chat-container');
    const conversationId = conversationContainer ? conversationContainer.dataset.conversationId : null;
    
    // Si le formulaire n'existe pas, on arrête
    if (!chatForm) return;
    
    // Ajustement automatique de la hauteur du textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    // Soumission du formulaire
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = messageInput.value.trim();
        
        if (message) {
            // Afficher le message de l'utilisateur
            addUserMessage(message);
            
            // Réinitialiser l'input
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // Afficher l'indicateur de saisie
            const typingIndicator = addTypingIndicator();
            
            // Envoyer la requête au serveur
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: message,
                    conversation_id: conversationId
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Supprimer l'indicateur de saisie
                typingIndicator.remove();
                
                // Afficher la réponse
                addAssistantMessage(data.response);
                
                // Mettre à jour l'URL avec le nouvel ID de conversation si nécessaire
                if (!conversationId) {
                    conversationContainer.dataset.conversationId = data.conversation_id;
                    history.pushState({}, '', `/chat/${data.conversation_id}`);
                }
                
                // Recharger les conversations récentes pour mettre à jour les titres
                loadRecentConversations();
            })
            .catch(error => {
                console.error('Error:', error);
                typingIndicator.remove();
                addAssistantMessage("Sorry, I couldn't process your request. Please try again.");
            });
        }
    });
}

/**
 * Ajoute un message utilisateur à la conversation
 */
function addUserMessage(message) {
    const chatMessages = document.getElementById('chatMessages');
    const template = document.getElementById('userMessageTemplate');
    
    if (!chatMessages || !template) return;
    
    const messageElement = template.content.cloneNode(true);
    messageElement.querySelector('p').textContent = message;
    chatMessages.appendChild(messageElement);
    
    // Scroll vers le bas
    scrollToBottom();
}

/**
 * Ajoute un message assistant à la conversation
 */
function addAssistantMessage(message) {
    const chatMessages = document.getElementById('chatMessages');
    const template = document.getElementById('assistantMessageTemplate');
    
    if (!chatMessages || !template) return;
    
    const messageElement = template.content.cloneNode(true);
    messageElement.querySelector('p').textContent = message;
    chatMessages.appendChild(messageElement);
    
    // Scroll vers le bas
    scrollToBottom();
}

/**
 * Ajoute l'indicateur de saisie
 */
function addTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const template = document.getElementById('typingIndicatorTemplate');
    
    if (!chatMessages || !template) return null;
    
    const indicator = template.content.cloneNode(true);
    chatMessages.appendChild(indicator);
    scrollToBottom();
    
    // Retourner l'élément pour pouvoir le supprimer plus tard
    return chatMessages.lastElementChild;
}

/**
 * Scroll vers le bas de la conversation
 */
function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}


// INTERFACE DE CHAT
document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const chatMessages = document.getElementById('chatMessages');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const chatContainer = document.querySelector('.chat-container');
    const conversationId = chatContainer.dataset.conversationId;

    // Définir le mode initial (welcome ou chat)
    function setInitialMode() {
        // Si une conversation existe et a des messages, activer le mode chat
        if (conversationId) {
            // Vérifier si la conversation a des messages
            fetch(`/api/conversations/${conversationId}/messages`)
                .then(response => response.json())
                .then(data => {
                    if (data.messages && data.messages.length > 0) {
                        // Cette conversation a des messages, donc afficher le mode chat
                        chatContainer.classList.add('chat-active');
                        
                        // Charger et afficher les messages
                        displayMessages(data.messages);
                    } else {
                        // Conversation vide ou nouvelle, afficher l'écran de bienvenue
                        chatContainer.classList.remove('chat-active');
                    }
                })
                .catch(error => {
                    console.error('Error loading messages:', error);
                    // En cas d'erreur, afficher l'écran de bienvenue par défaut
                    chatContainer.classList.remove('chat-active');
                });
        } else {
            // Pas d'ID de conversation, afficher l'écran de bienvenue
            chatContainer.classList.remove('chat-active');
        }
    }

    // Fonction pour afficher les messages depuis l'API
    function displayMessages(messages) {
        // Vider d'abord le conteneur de messages
        chatMessages.innerHTML = '';
        
        // Afficher chaque message
        messages.forEach(msg => {
            if (msg.role === 'user') {
                addUserMessage(msg.content);
            } else if (msg.role === 'assistant') {
                addAssistantMessage(msg.content);
            }
        });
    }
    
    // Ajustement automatique de la hauteur du textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    });
    
    // Soumission du formulaire
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = messageInput.value.trim();
        
        if (message) {
            // Activer le mode chat (cache le welcome screen)
            chatContainer.classList.add('chat-active');
            
            // Afficher le message de l'utilisateur
            addUserMessage(message);
            
            // Réinitialiser l'input
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // Afficher l'indicateur de saisie
            const typingIndicator = addTypingIndicator();
            
            // Envoyer la requête au serveur
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: message,
                    conversation_id: conversationId || '' // S'assurer que c'est une chaîne vide si null
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Supprimer l'indicateur de saisie
                typingIndicator.remove();
                
                // Afficher la réponse
                addAssistantMessage(data.response);
                
                // Mettre à jour l'URL avec le nouvel ID de conversation si nécessaire
                if (window.location.pathname.endsWith('/chat')) {
                    // Mettre à jour l'URL
                    history.pushState({}, '', `/chat/${data.conversation_id}`);
                    
                    // Mettre à jour l'ID de conversation pour les futurs messages
                    chatContainer.dataset.conversationId = data.conversation_id;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                typingIndicator.remove();
                addAssistantMessage("Sorry, I couldn't process your request. Please try again.");
            });
        }
    });
    
    // Fonction pour ajouter un message utilisateur
    function addUserMessage(message) {
        const template = document.getElementById('userMessageTemplate');
        const messageElement = template.content.cloneNode(true);
        
        messageElement.querySelector('p').textContent = message;
        chatMessages.appendChild(messageElement);
        
        // Activer le bouton d'édition
        const editButton = chatMessages.lastElementChild.querySelector('.edit-message');
        if (editButton) {
            editButton.addEventListener('click', function() {
                // Mettre le message dans l'input
                messageInput.value = message;
                messageInput.focus();
                messageInput.style.height = 'auto';
                messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
                
                // Faire défiler jusqu'à la zone de saisie
                messageInput.scrollIntoView({ behavior: 'smooth' });
            });
        }
        
        // Scroll vers le bas
        scrollToBottom();
    }
    
    // Fonction pour ajouter un message assistant
    function addAssistantMessage(message) {
        const template = document.getElementById('assistantMessageTemplate');
        const messageElement = template.content.cloneNode(true);
        
        messageElement.querySelector('p').textContent = message;
        chatMessages.appendChild(messageElement);
        
        // Activer le bouton de copie
        const copyButton = chatMessages.lastElementChild.querySelector('.copy-message');
        if (copyButton) {
            copyButton.addEventListener('click', function() {
                navigator.clipboard.writeText(message)
                    .then(() => {
                        // Feedback visuel temporaire
                        const originalHTML = this.innerHTML;
                        this.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#333" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
                        
                        setTimeout(() => {
                            this.innerHTML = originalHTML;
                        }, 2000);
                    })
                    .catch(err => {
                        console.error('Could not copy text: ', err);
                    });
            });
        }
        
        // Scroll vers le bas
        scrollToBottom();
    }
    
    // Fonction pour ajouter l'indicateur de saisie
    function addTypingIndicator() {
        const template = document.getElementById('typingIndicatorTemplate');
        const indicator = template.content.cloneNode(true);
        
        chatMessages.appendChild(indicator);
        scrollToBottom();
        
        // Retourner l'élément pour pouvoir le supprimer plus tard
        return chatMessages.lastElementChild;
    }
    
    // Fonction pour scroller vers le bas
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Permettre l'envoi avec Entrée, mais Shift+Entrée pour nouvelle ligne
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Empêcher le saut de ligne
            chatForm.dispatchEvent(new Event('submit')); // Simuler un clic sur "Envoyer"
        }
    });
    
    // Observer les changements d'URL pour réinitialiser le mode quand on navigue vers /chat sans ID
    function handleURLChange() {
        if (window.location.pathname === '/chat') {
            // Réinitialiser l'ID de conversation
            chatContainer.dataset.conversationId = '';
            
            // Réinitialiser le mode
            chatContainer.classList.remove('chat-active');
            
            // Vider les messages
            chatMessages.innerHTML = '';
        }
    }
    
    // Écouter les changements d'historique (bouton retour/avant)
    window.addEventListener('popstate', handleURLChange);
    
    // Initialiser le mode au chargement
    setInitialMode();
});