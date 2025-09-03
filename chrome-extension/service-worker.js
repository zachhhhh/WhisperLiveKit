console.log("Service worker loaded");

let isRecording = false;
let currentStreamId = null;

chrome.runtime.onInstalled.addListener((details) => {
  console.log("Extension installed/updated");
});

chrome.action.onClicked.addListener((tab) => {
  // Get the current tab ID
  const tabId = tab.id;
  
  // Inject the content script into the current tab
  chrome.scripting.executeScript({
    target: { tabId: tabId },
    files: ['style_popup.js']
  });
  
  console.log(`Content script injected into tab ${tabId}`);
}); 


// Handle messages from popup
chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
  console.log("Service worker received message:", message);
  
  try {
    switch (message.type) {
      case 'start-capture':
        const startResult = await startTabCapture(message.tabId, message.websocketUrl);
        sendResponse(startResult);
        break;
        
      case 'stop-capture':
        const stopResult = await stopTabCapture();
        sendResponse(stopResult);
        break;
        
      case 'get-recording-state':
        sendResponse({ isRecording: isRecording });
        break;
        
      default:
        sendResponse({ success: false, error: 'Unknown message type' });
    }
  } catch (error) {
    console.error('Error handling message:', error);
    sendResponse({ success: false, error: error.message });
  }
  
  return true; // Keep message channel open for async response
});

async function startTabCapture(tabId, websocketUrl) {
  console.log('Service worker: Starting tab capture process...');
  console.log('Service worker: tabId:', tabId, 'websocketUrl:', websocketUrl);

  try {
    if (isRecording) {
      console.log('Service worker: Already recording, aborting');
      return { success: false, error: 'Already recording' };
    }

    // Check if offscreen document exists
    console.log('Service worker: Checking for existing offscreen document...');
    const existingContexts = await chrome.runtime.getContexts({});
    console.log('Service worker: Found contexts:', existingContexts.length);

    const offscreenDocument = existingContexts.find(
      (c) => c.contextType === 'OFFSCREEN_DOCUMENT'
    );

    console.log('Service worker: Offscreen document exists:', !!offscreenDocument);

    // Create offscreen document if it doesn't exist
    if (!offscreenDocument) {
      console.log('Service worker: Creating offscreen document...');
      try {
        await chrome.offscreen.createDocument({
          url: 'offscreen.html',
          reasons: ['USER_MEDIA'],
          justification: 'Capturing and processing tab audio for transcription'
        });
        console.log('Service worker: Offscreen document created successfully');

        // Wait for offscreen document to initialize
        console.log('Service worker: Waiting for offscreen document to initialize...');
        await new Promise(resolve => setTimeout(resolve, 500));
        console.log('Service worker: Offscreen document initialization delay complete');

      } catch (offscreenError) {
        console.error('Service worker: Failed to create offscreen document:', offscreenError);
        return { success: false, error: 'Failed to create offscreen document: ' + offscreenError.message };
      }
    }

    // Get media stream ID for the tab
    console.log('Service worker: Getting media stream ID for tab:', tabId);
    try {
      currentStreamId = await chrome.tabCapture.getMediaStreamId({
        targetTabId: tabId
      });
      console.log('Service worker: Media stream ID:', currentStreamId);
    } catch (tabCaptureError) {
      console.error('Service worker: Failed to get media stream ID:', tabCaptureError);
      return { success: false, error: 'Failed to get media stream ID: ' + tabCaptureError.message };
    }

    if (!currentStreamId) {
      console.log('Service worker: No media stream ID returned');
      return { success: false, error: 'Failed to get media stream ID - no stream returned' };
    }

    // Send message to offscreen document to start capture with retry logic
    console.log('Service worker: Sending start message to offscreen document...');

    let response;
    let retryCount = 0;
    const maxRetries = 5;

    while (!response && retryCount < maxRetries) {
      try {
        console.log(`Service worker: Attempt ${retryCount + 1}/${maxRetries} to communicate with offscreen document`);

        // Send message to offscreen document without target property
        response = await chrome.runtime.sendMessage({
          type: 'start-recording',
          target: 'offscreen',
          data: {
            streamId: currentStreamId,
            websocketUrl: websocketUrl
          }
        });

        if (!response) {
          console.warn(`Service worker: No response from offscreen document, waiting before retry...`);
          await new Promise(resolve => setTimeout(resolve, 200));
          retryCount++;
        } else {
          console.log(`Service worker: Successfully communicated with offscreen document on attempt ${retryCount + 1}`);
        }
      } catch (sendError) {
        console.error(`Service worker: Error sending message to offscreen document (attempt ${retryCount + 1}):`, sendError);
        response = { success: false, error: 'Failed to communicate with offscreen document: ' + sendError.message };
        break;
      }
    }

    console.log('Service worker: Final offscreen document response:', response);

    if (response && response.success) {
      isRecording = true;
      console.log('Service worker: Recording started successfully');

      // Notify popup of state change
      try {
        chrome.runtime.sendMessage({
          type: 'recording-state',
          isRecording: true
        });
      } catch (e) {
        console.warn('Service worker: Could not notify popup of state change:', e);
      }

      return { success: true };
    } else {
      console.log('Service worker: Offscreen document returned failure');
      return { success: false, error: response?.error || 'Failed to start recording in offscreen document' };
    }

  } catch (error) {
    console.error('Service worker: Exception in startTabCapture:', error);
    return { success: false, error: 'Exception: ' + error.message };
  }
}

async function stopTabCapture() {
  try {
    if (!isRecording) {
      return { success: false, error: 'Not currently recording' };
    }
    
    // Send message to offscreen document to stop capture
    const response = await chrome.runtime.sendMessage({
      type: 'stop-recording',
      target: 'offscreen'
    });
    
    isRecording = false;
    currentStreamId = null;
    
    // Notify popup of state change
    try {
      chrome.runtime.sendMessage({
        type: 'recording-state',
        isRecording: false
      });
    } catch (e) {
      // Popup might be closed, ignore error
    }
    
    return { success: true };
    
  } catch (error) {
    console.error('Error stopping tab capture:', error);
    isRecording = false;
    currentStreamId = null;
    return { success: false, error: error.message };
  }
}

// Handle messages from offscreen document
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.target === 'service-worker') {
    switch (message.type) {
      case 'recording-stopped':
        isRecording = false;
        currentStreamId = null;
        
        // Notify popup
        try {
          chrome.runtime.sendMessage({
            type: 'recording-state',
            isRecording: false
          });
        } catch (e) {
          // Popup might be closed, ignore error
        }
        break;
        
      case 'recording-error':
        isRecording = false;
        currentStreamId = null;
        
        // Notify popup
        try {
          chrome.runtime.sendMessage({
            type: 'status-update',
            status: 'error',
            message: message.error || 'Recording error occurred'
          });
        } catch (e) {
          // Popup might be closed, ignore error
        }
        break;
    }
  }
});
