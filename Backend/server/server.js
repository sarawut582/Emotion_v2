const express = require('express')
const app = express()
const morgan = require('morgan')
const bodyParser = require('body-parser')
const cors = require('cors')

// Import routes



app.use(morgan('dev'))
app.use(bodyParser.json())
app.use(cors())

// Routes


// Admin

app.listen(5000 , () => console.log('server is Running on port 5000'))