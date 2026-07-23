using System.Diagnostics;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
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

		// カメラから取得した映像: PixelFormats.Bgr24
		// 背景を除去した映像: PixelFormats.Bgra32
		public WriteableBitmap Bitmap { get; } = new WriteableBitmap(CameraWidth, CameraHeight, 96, 96, PixelFormats.Bgra32, null);

		readonly Dispatcher uiDispatcher;
		bool isOn = true;

		public AppModel()
		{
			uiDispatcher = Dispatcher.CurrentDispatcher;
			Task.Delay(200)
				.ContinueWith(_ => CaptureCamera());
		}

		void CaptureCamera()
		{
			if (IsInDesignMode) return;

			using var segmentation = new SelfieSegmentationModel("./Assets/selfie_segmentation.onnx");
			using var capture = new VideoCapture(CameraIndex, VideoCaptureAPIs.DSHOW);

			// 既定で 640 x 480
			// FPS の設定は反映されないようです。
			//capture.FrameWidth = CameraWidth;
			//capture.FrameHeight = CameraHeight;
			//capture.Fps = CameraFps;

			if (!capture.IsOpened()) return;

			using var frame = new Mat();
			var frameRate = new FrameRateManager(CameraFps);

			while (isOn)
			{
				if (!capture.Read(frame) || frame.Empty()) continue;

				if (!frameRate.IsAvailable()) continue;
				Debug.WriteLine($"FPS: {frameRate.ActualFps}");

				var bgra = segmentation.RemoveBackground(frame, MaskThreshold);
				uiDispatcher.Invoke(() => UpdateImage(bgra));
			}
		}

		void UpdateImage(Mat frame)
		{
			Bitmap.WritePixels(
				new Int32Rect(0, 0, frame.Width, frame.Height),
				frame.Data,
				frame.Height * (int)frame.Step(),
				(int)frame.Step()
			);
		}

		public void Close()
		{
			isOn = false;
			Task.Delay(200).Wait();
		}
	}
}
