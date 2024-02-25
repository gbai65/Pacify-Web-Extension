var blockButton = document.getElementById('block-button');
blockButton.addEventListener('click', function() {
    window.location.reload();
    //document.querySelector('block-button').value = 'Blocked';
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        //chrome.runtime.sendMessage({ action: "execute_main_js" });
        chrome.tabs.sendMessage(tabs[0].id,{
            s:"helloworld"
        })
    });
});
