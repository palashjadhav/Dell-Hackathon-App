
function showDiv(){
	var c = document.getElementById("chcYes");
	var inp = document.getElementById("inpField");

	inp.style.display = c.checked ? "block" : "none";
}
function sendMail(){
	var temp = document.getElementById("mailTemp");
	var temp1 = document.getElementById("confID");
	temp.style.display = "block";
	temp1.style.display = "block";
}
function removeDiv(){
	var temp = document.getElementById("mailTemp");
	var temp1 = document.getElementById("confID");
	temp.style.display = "none";
	temp1.style.display = "none";
}