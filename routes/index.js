var express = require('express');
var router = express.Router();
const multer = require('multer')
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

function ImgBufferToHtml(bufferArray) {
  const base64Image = bufferArray.toString('base64');
  const htmlFormatImg = "data:image/png;base64," + base64Image
  return htmlFormatImg;
}

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Cat Identification' });
});

router.post('/myaction', upload.single('userImage'), async function (req, res, next) {
  if (!req.file) {
    return res.status(400).send('No file uploaded.');
  }
  const selectedBreed = req.body.selectedBreed;
  console.log(req.file.buffer);
  const userImage = req.file.buffer;
  fetch("http://127.0.0.1:5000/inference", {
    method: "POST",
    headers: { 'Content-Type': 'application/json' },
    body: userImage
  }).then(response => response.json())  // Use a different variable name here
    .then(data => {
      images = data.images
      labels = data.labels
      const image_to_show = [];
      for (let i = 0; i < labels.length; i++) {
        image_to_show.push(ImgBufferToHtml(images[i]));
      }
      res.render('result', { title: 'Cat Identification', selectedBreed, image_to_show });
    });
});



module.exports = router;


