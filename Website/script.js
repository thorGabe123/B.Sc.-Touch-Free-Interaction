const keys = document.querySelectorAll('.key');
const output = document.getElementById("output");
var socket = new WebSocket('ws://localhost:8000');
        
keys.forEach(key => key.addEventListener('click', () => {
  console.log(key.innerHTML);
  output.innerText += key.innerText;
}));

socket.onmessage = function(event) {
    var message = event.data;
    document.getElementById('message').innerHTML = message;
}