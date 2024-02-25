chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
    try{
        var elements = document.querySelectorAll('body p, body h1, body h2, body h3, body h4, body h5, body h6, body span, body a');
        elements.forEach(element => {
            const textContent = element.textContent;
            if (textContent !== null) {
                const regex = new RegExp('hate speech', 'gi');
                if (textContent.match(regex)) {
                    element.innerHTML = element.innerHTML.replace(regex, '<span style="background-color: yellow; color: black">*censored*</span>');
                    if (element.tagName.toLowerCase() === 'a') 
                        element.setAttribute('href', element.getAttribute('href'));
                }
            }  
    
        });
        const button = document.getElementById('block-button');
                button.textContent = 'Blocked!';
    }    
    catch (error){
        console.log(error);
    }
    });
    
    
