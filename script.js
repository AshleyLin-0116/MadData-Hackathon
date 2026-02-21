const input = document.getElementById('taskInput');
const btn = document.getElementById('addBtn');
const list = document.getElementById('taskList');

btn.onclick = () => {
    if (!input.value) return;
    
    const li = document.createElement('li');
    li.innerHTML = `
        <span>${input.value}</span>
        <button onclick="this.parentElement.remove()">Delete</button>
    `;
    list.appendChild(li);
    input.value = '';
};