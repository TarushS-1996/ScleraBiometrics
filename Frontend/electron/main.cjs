// const { app, BrowserWindow, session } = require('electron')
// const path = require('path')

// const isDev = !app.isPackaged

// async function createWindow() {

//   // ASK CAMERA PERMISSION ON MAC
//   if (process.platform === "darwin") {
//     const status = systemPreferences.getMediaAccessStatus("camera");

//     if (status !== "granted") {
//       await systemPreferences.askForMediaAccess("camera");
//     }
//   }
//   const win = new BrowserWindow({
//     width: 1200,
//     height: 800,
//     webPreferences: {
//       preload: path.join(__dirname, 'preload.js'),
//     },
//   })

//   // Allow camera/microphone permissions
//   session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback) => {
//     const allowed = ['media', 'mediaKeySystem'];
//     callback(allowed.includes(permission));
//   });

//   if (isDev) {
//     win.loadURL('http://localhost:5173')
//   } else {
//     win.loadFile(path.join(__dirname, '../dist/index.html'))
//   }
// }

// app.whenReady().then(createWindow)

// app.on('window-all-closed', () => {
//   if (process.platform !== 'darwin') app.quit()
// })


const {
  app,
  BrowserWindow,
  session,
  systemPreferences,
} = require('electron')

const path = require('path')

const isDev = !app.isPackaged

async function createWindow() {

  // MAC CAMERA PERMISSION
  if (process.platform === 'darwin') {
    const status = systemPreferences.getMediaAccessStatus('camera')

    if (status !== 'granted') {
      await systemPreferences.askForMediaAccess('camera')
    }
  }

  // ALLOW MEDIA PERMISSIONS
  session.defaultSession.setPermissionRequestHandler(
    (_webContents, permission, callback) => {
      const allowedPermissions = ['media']

      callback(allowedPermissions.includes(permission))
    }
  )

  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  if (isDev) {
    win.loadURL('http://localhost:5173')
  } else {
    win.loadFile(path.join(__dirname, '../dist/index.html'))
  }
}

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})