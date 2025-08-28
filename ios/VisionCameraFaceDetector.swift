import Vision
import VisionCamera
import UIKit
import AVFoundation
import CoreML
import CoreGraphics
import CoreImage

@objc(VisionCameraFaceDetector)
public class VisionCameraFaceDetector: FrameProcessorPlugin {

  // MARK: - Init

  @objc
  public override init(proxy: VisionCameraProxyHolder, options: [AnyHashable: Any]! = [:]) {
    super.init(proxy: proxy, options: options)
  }

  // MARK: - Utilities

  private func degrees(from radians: Double?) -> Double? {
    guard let radians else { return nil }
    return radians * 180.0 / .pi
  }

  private func denormalizeRect(_ rect: CGRect, imgW: Int, imgH: Int) -> CGRect {
    let screenW = UIScreen.main.bounds.width
    let screenH = UIScreen.main.bounds.height
    let isPortrait = imgH > imgW
    let imageHeight = isPortrait ? imgH : imgW
    let imageWidth = isPortrait ? imgW : imgH
    let virtualH = screenH
    let virtualW = CGFloat(imageWidth) / CGFloat(imageHeight) * virtualH
    let scale = virtualH / CGFloat(imageHeight)
    let offsetX = (screenW - virtualW) / 2

    let xImg = rect.origin.y * CGFloat(imageWidth)
    let yImg = (1.0 - rect.origin.x - rect.height) * CGFloat(imageHeight)
    let wImg = rect.width * CGFloat(imgW)
    let hImg = rect.height * CGFloat(imgH)

    let x = xImg * scale + offsetX
    let y = (CGFloat(imageHeight) - yImg - (rect.height * CGFloat(imageHeight))) * scale
    let width = (wImg * scale)
    let height = (hImg * scale)

    return CGRect(x: x, y: y, width: width, height: height)
  }

  private func landmarkPoints(
    from region: VNFaceLandmarkRegion2D?,
    faceBBox: CGRect,
    imgW: Int, imgH: Int
  ) -> [[String: CGFloat]] {
    guard let region else { return [] }

    return region.normalizedPoints.map { p in
      let nx = faceBBox.origin.x + CGFloat(p.x) * faceBBox.width
      let ny = faceBBox.origin.y + CGFloat(p.y) * faceBBox.height
      return [
        "x": nx * CGFloat(imgW),
        "y": (1.0 - ny) * CGFloat(imgH)
      ]
    }
  }

  private func boundingBoxDict(_ rect: CGRect) -> [String: Any] {
    [
      "x": rect.origin.x,
      "y": rect.origin.y,
      "width": rect.width,
      "height": rect.height,
      "boundingCenterX": rect.midX,
      "boundingCenterY": rect.midY
    ]
  }

  private func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
    hypot(p1.x - p2.x, p1.y - p2.y)
  }

  private func averagePoint(_ points: [CGPoint]) -> CGPoint {
    guard !points.isEmpty else { return .zero }
    let sum = points.reduce(.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
    return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
  }

  // MARK: - Eye EAR utilities

  private func pointFromNormalized(_ p: CGPoint, faceBBox: CGRect, imgW: Int, imgH: Int) -> CGPoint {
    let nx = faceBBox.origin.x + p.x * faceBBox.width
    let ny = faceBBox.origin.y + p.y * faceBBox.height
    return CGPoint(x: nx * CGFloat(imgW), y: (1.0 - ny) * CGFloat(imgH))
  }

  /// Tính EAR từ 6 điểm mắt
  private func computeEAR(from region: VNFaceLandmarkRegion2D?, faceBBox: CGRect, imgW: Int, imgH: Int) -> CGFloat? {
    guard let region = region, region.pointCount >= 6 else { return nil }

    let pts = region.normalizedPoints
    let p1 = pointFromNormalized(pts[0], faceBBox: faceBBox, imgW: imgW, imgH: imgH)
    let p2 = pointFromNormalized(pts[1], faceBBox: faceBBox, imgW: imgW, imgH: imgH)
    let p3 = pointFromNormalized(pts[2], faceBBox: faceBBox, imgW: imgW, imgH: imgH)
    let p4 = pointFromNormalized(pts[3], faceBBox: faceBBox, imgW: imgW, imgH: imgH)
    let p5 = pointFromNormalized(pts[4], faceBBox: faceBBox, imgW: imgW, imgH: imgH)
    let p6 = pointFromNormalized(pts[5], faceBBox: faceBBox, imgW: imgW, imgH: imgH)

    let vert1 = distance(p2, p6)
    let vert2 = distance(p3, p5)
    let horiz = distance(p1, p4)
    guard horiz > 0 else { return nil }
    return (vert1 + vert2) / (2.0 * horiz)
  }

  private func earToProbability(ear: CGFloat, baselineOpenEAR: CGFloat? = nil) -> CGFloat {
    if let base = baselineOpenEAR, base > 0 {
      let p = ear / base
      return min(max(p, 0), 1)
    } else {
      let minEAR: CGFloat = 0.08
      let maxEAR: CGFloat = 0.32
      return min(max((ear - minEAR) / (maxEAR - minEAR), 0), 1)
    }
  }

  private var leftEyeEMA: CGFloat = 0.0
  private var rightEyeEMA: CGFloat = 0.0
  private let emaAlpha: CGFloat = 0.3

  private func smoothEMA(previous: CGFloat, newValue: CGFloat) -> CGFloat {
    return previous * (1.0 - emaAlpha) + newValue * emaAlpha
  }

  private func eyeState(from prob: CGFloat) -> String {
    if prob < 0.2 { return "CLOSED" }
    if prob > 0.45 { return "OPEN" }
    return "UNKNOWN"
  }

  // MARK: - Smile + Pitch + Brightness

  private func estimateSmile(lips: VNFaceLandmarkRegion2D?) -> CGFloat? {
    guard let lips, lips.pointCount > 5 else { return nil }

    let points = lips.normalizedPoints
    let leftCorner = points.first!
    let rightCorner = points[points.count / 2]
    let topMid = points[points.count / 4]
    let bottomMid = points[3 * points.count / 4]

    let width = distance(CGPoint(x: CGFloat(leftCorner.x), y: CGFloat(leftCorner.y)),
                         CGPoint(x: CGFloat(rightCorner.x), y: CGFloat(rightCorner.y)))
    let height = distance(CGPoint(x: CGFloat(topMid.x), y: CGFloat(topMid.y)),
                          CGPoint(x: CGFloat(bottomMid.x), y: CGFloat(bottomMid.y)))

    guard width > 0 else { return nil }
    let ratio = height / width

    return min(max((ratio - 0.2) / 0.3, 0), 1)
  }

  private func estimatePitch(face: VNFaceObservation) -> Float {
    guard let landmarks = face.landmarks,
          let leftEye = landmarks.leftEye?.normalizedPoints,
          let rightEye = landmarks.rightEye?.normalizedPoints,
          let mouth = landmarks.outerLips?.normalizedPoints else {
      return 0.0
    }

    let eyeCenterY = (averagePoint(leftEye).y + averagePoint(rightEye).y) / 2.0
    let mouthCenterY = averagePoint(mouth).y
    let eyeMouthDist = abs(mouthCenterY - eyeCenterY)
    let ratio = Float(eyeMouthDist / face.boundingBox.height)

    return (0.5 - ratio) * 60.0
  }

  private func estimateBrightness(from sampleBuffer: CMSampleBuffer) -> Float {
    guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return 0 }

    let ciImage = CIImage(cvPixelBuffer: imageBuffer)
    let filter = CIFilter(
      name: "CIAreaAverage",
      parameters: [
        kCIInputImageKey: ciImage,
        kCIInputExtentKey: CIVector(cgRect: ciImage.extent)
      ]
    )

    guard let outputImage = filter?.outputImage else { return 0 }

    var bitmap = [UInt8](repeating: 0, count: 4)
    CIContext(options: [.workingColorSpace: NSNull()]).render(
      outputImage,
      toBitmap: &bitmap,
      rowBytes: 4,
      bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
      format: .RGBA8,
      colorSpace: nil
    )

    let (r, g, b) = (Float(bitmap[0]), Float(bitmap[1]), Float(bitmap[2]))
    return ((r + g + b) / (3 * 255.0)) * 100
  }

  // MARK: - Callback

  @objc
  public override func callback(_ frame: Frame, withArguments arguments: [AnyHashable: Any]?) -> Any? {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(frame.buffer) else { return nil }

    let imgW = CVPixelBufferGetWidth(pixelBuffer)
    let imgH = CVPixelBufferGetHeight(pixelBuffer)
    var faceAttributes: [[String: Any]] = []

    let request = VNDetectFaceLandmarksRequest { req, err in
      guard err == nil, let faces = req.results as? [VNFaceObservation], !faces.isEmpty else { return }

      for face in faces {
        let bbox = self.denormalizeRect(face.boundingBox, imgW: imgW, imgH: imgH)
        let landmarks = face.landmarks
        var contours: [String: [[String: CGFloat]]] = [:]

        func addContour(_ key: String, _ region: VNFaceLandmarkRegion2D?) {
          contours[key] = self.landmarkPoints(from: region, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        }

        addContour("FACE", landmarks?.faceContour)
        addContour("LEFT_EYEBROW_TOP", landmarks?.leftEyebrow)
        contours["LEFT_EYEBROW_BOTTOM"] = contours["LEFT_EYEBROW_TOP"]
        addContour("RIGHT_EYEBROW_TOP", landmarks?.rightEyebrow)
        contours["RIGHT_EYEBROW_BOTTOM"] = contours["RIGHT_EYEBROW_TOP"]
        addContour("LEFT_EYE", landmarks?.leftEye)
        addContour("RIGHT_EYE", landmarks?.rightEye)
        addContour("NOSE_BRIDGE", landmarks?.noseCrest)
        addContour("NOSE_BOTTOM", landmarks?.nose)
        addContour("UPPER_LIP_TOP", landmarks?.outerLips)
        addContour("UPPER_LIP_BOTTOM", landmarks?.innerLips)
        contours["LOWER_LIP_TOP"] = contours["UPPER_LIP_BOTTOM"]
        contours["LOWER_LIP_BOTTOM"] = contours["UPPER_LIP_TOP"]

        let brightness = self.estimateBrightness(from: frame.buffer)

        // Eye EAR cho 2 mắt
        var leftEyeProb: CGFloat = 0
        var rightEyeProb: CGFloat = 0
        var leftEyeState = "UNKNOWN"
        var rightEyeState = "UNKNOWN"

        if let leftEAR = self.computeEAR(from: landmarks?.leftEye, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH) {
          let probRaw = self.earToProbability(ear: leftEAR)
          self.leftEyeEMA = self.smoothEMA(previous: self.leftEyeEMA, newValue: probRaw)
          leftEyeProb = self.leftEyeEMA
          leftEyeState = self.eyeState(from: leftEyeProb)
        }

        if let rightEAR = self.computeEAR(from: landmarks?.rightEye, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH) {
          let probRaw = self.earToProbability(ear: rightEAR)
          self.rightEyeEMA = self.smoothEMA(previous: self.rightEyeEMA, newValue: probRaw)
          rightEyeProb = self.rightEyeEMA
          rightEyeState = self.eyeState(from: rightEyeProb)
        }

        var map: [String: Any] = [
          "rollAngle": self.degrees(from: face.roll?.doubleValue) ?? NSNull(),
          "pitchAngle": self.estimatePitch(face: face),
          "yawAngle": self.degrees(from: face.yaw?.doubleValue) ?? NSNull(),
          "bounds": self.boundingBoxDict(bbox),
          "contours": contours,
          "brightness": brightness,
          "leftEyeOpenProbability": leftEyeProb,
          "rightEyeOpenProbability": rightEyeProb,
          "leftEyeState": leftEyeState,
          "rightEyeState": rightEyeState,
          "smilingProbability": self.estimateSmile(lips: landmarks?.outerLips) ?? NSNull()
        ]

        faceAttributes.append(map)
      }
    }

    do {
      if #available(iOS 14.0, *) {
        let handler = VNImageRequestHandler(cmSampleBuffer: frame.buffer,
                                            orientation: .up,
                                            options: [:])
        try handler.perform([request])
      } else {
        if let pixelBuffer = CMSampleBufferGetImageBuffer(frame.buffer) {
          let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                              orientation: .up,
                                              options: [:])
          try handler.perform([request])
        }
      }
    } catch {
      return nil
    }

    return faceAttributes
  }
}
