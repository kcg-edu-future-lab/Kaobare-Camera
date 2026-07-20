using System.Diagnostics;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using OpenCvSharp;

namespace KaobareCamera
{
	public class AppModel
	{
		const int CameraIndex = 0;
		const int CameraWidth = 640;
		const int CameraHeight = 480;
		const int CameraFps = 30;
		const double MaskThreshold = 0.35;

		public bool IsInDesignMode { get; set; }
		public WriteableBitmap Bitmap { get; } = new WriteableBitmap(CameraWidth, CameraHeight, 96, 96, PixelFormats.Bgr24, null);

		public AppModel()
		{
			Task.Run(CaptureCamera);
		}

		void CaptureCamera()
		{
			if (IsInDesignMode) return;

			using var capture = new VideoCapture(CameraIndex, VideoCaptureAPIs.DSHOW);
			capture.Set(VideoCaptureProperties.FrameWidth, CameraWidth);
			capture.Set(VideoCaptureProperties.FrameHeight, CameraHeight);
			capture.Set(VideoCaptureProperties.Fps, CameraFps);

			if (!capture.IsOpened()) return;

			using var frame = new Mat();
			var dt = DateTime.Now;
			while (true)
			{
				if (!capture.Read(frame) || frame.Empty()) continue;

				var fps = 1 / (DateTime.Now - dt).TotalSeconds;
				Debug.WriteLine($"FPS: {fps}");
				dt = DateTime.Now;

				Application.Current.Dispatcher.Invoke(() => UpdateOriginalImage(frame));
			}
		}

		void UpdateOriginalImage(Mat frame)
		{
			Bitmap.WritePixels(
				new Int32Rect(0, 0, frame.Width, frame.Height),
				frame.Data,
				frame.Height * (int)frame.Step(),
				(int)frame.Step()
			);
		}
	}
}
