let print = console.log;

function indexOfMax(arr) {
	if (arr.length === 0) {
		return -1;
	}

	let max = arr[0];
	let maxIndex = 0;

	for (var i = 1; i < arr.length; i++) {
		if (arr[i] > max) {
			maxIndex = i;
			max = arr[i];
		}
	}

	return maxIndex;
}

async function loadModel() {
	window.model = await tf.loadLayersModel("model.json");
}
loadModel();

window.onload = function () {
	let canvas = document.getElementById("digitCanvas");
	let context = canvas.getContext("2d");
	let boundings = canvas.getBoundingClientRect();
	let mouseX = 0;
	let mouseY = 0;
	let isDrawing = false;

	context.strokeStyle = "black"; // initial brush color
	context.lineWidth = 16; // initial brush width
	// Mouse Down Event
	canvas.addEventListener("mousedown", function(event) {
		setMouseCoordinates(event);
		isDrawing = true;

		// Start Drawing
		context.beginPath();
		context.moveTo(mouseX, mouseY);
	});

	// Mouse Move Event
	canvas.addEventListener("mousemove", function(event) {
		setMouseCoordinates(event);

		if(isDrawing) {
		  context.lineTo(mouseX, mouseY);
		  context.stroke();
		}
	});

	// Mouse Up Event
	canvas.addEventListener("mouseup", function(event) {
		setMouseCoordinates(event);
		isDrawing = false;
	});

	canvas.addEventListener("mouseout", function(event) {
		isDrawing = false;
	});

	// Handle Mouse Coordinates
	function setMouseCoordinates(event) {
		mouseX = event.clientX - boundings.left;
		mouseY = event.clientY - boundings.top;
	}

	// Handle Clear Button
	let clearButton = document.getElementById("clearButton");

	clearButton.addEventListener("click", function() {
		context.clearRect(0, 0, canvas.width, canvas.height);
	});
  
	// Handle Submit Button
	let submitButton = document.getElementById("submitButton")

	// Linear interpolation
	function lerp(start, end, perc) {
		return start + (end - start) * perc;
	}

	submitButton.addEventListener("click", function() {
		// Get pixel data from image
		let rowLength = canvas.width;

		let resizeCanvas = document.createElement("canvas");
		resizeCanvas.width = 28;
		resizeCanvas.height = 28;
		resizeCanvas.getContext("2d").drawImage(canvas, 0, 0, 28, 28);

		let imageData = resizeCanvas.getContext("2d").getImageData(0, 0, rowLength, rowLength).data;
		
		
		// Create array for the processed image data
		let processedImageData = [];
		
		// Stores the current row (immediately incremented so set to -1)
		let row = -1;


		// Set to 1 if pixel is drawn 0 otherwise
		for(let imageDataArrayIndex = 0; imageDataArrayIndex < imageData.length / 10; imageDataArrayIndex += 4) {
			let pixelNumber = imageDataArrayIndex / 4;

			// If it's a new row of pixels then increment row
			if (pixelNumber % rowLength == 0) {
				row = row + 1;
			}

			if (pixelNumber % rowLength < 28) {
				// If canvas alpha is not 0 then a pixel is drawn otherwise it's not
				if (imageData[imageDataArrayIndex + 3] != 0) {
					// If the current row already has an array then store it in the premade array
					if (Array.isArray(processedImageData[row])) {
						processedImageData[row][pixelNumber % rowLength] = 1;
					} else {
						// Otherwise create an array and add the value
						processedImageData[row] = [1];
					}
				} else {
					if (Array.isArray(processedImageData[row])) {
						// If the current row already has an array then store it in the premade array
						processedImageData[row][pixelNumber % rowLength] = 0;
					} else {
						// Otherwise create an array and add the value
						processedImageData[row] = [0];
					}
				}
			}
		}
		
		let pixelArray = [];
		let outputDiv = document.getElementById("outputDiv")
		outputDiv.innerHTML = "";
		processedImageData.forEach(function(element) {
			element.forEach(function(pixel, index) {
				if(index < 28) {
					pixelArray.push(pixel);
				}
			});
			//outputDiv.innerHTML = outputDiv.innerHTML + element.join("") + "<br>";
		});

		let pixelInput = [];
		pixelInput.push(pixelArray);
		let tensor = tf.tensor(pixelInput, [1, 784]);
		tensor.reshape([1, 28, 28, 1]);

		let prediction = window.model.predict(tensor);

		prediction.print()

		let dataPromise = prediction.data();
		dataPromise.then((value) => {
			let predictedNumber = indexOfMax(value);
			outputDiv.innerHTML = outputDiv.innerHTML + `<br><br><br><h2>Oioi savoi is that a ${predictedNumber}</h2>`
		});

		//outputDiv.innerHTMLasd

	});
};

