class Hivebear < Formula
  desc "Run local AI models, with picks matched to your hardware"
  homepage "https://hivebear.com"
  version "0.1.11"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-apple-darwin.tar.gz"
      sha256 "56138ed546da13b445514f8dbc80981f8945d9fdcafbd9e9a12930eb7d603b14"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-apple-darwin.tar.gz"
      sha256 "bed9815106b5b61fd05961655f13dd9287eadc63d0207bd7074d5b227b30994b"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "4d674643860cf3e7b2868ac7f67c7195642d4c4e31c8406043db8236ed6b7bed"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "f64540abb344c5332a9b44fce38df9c6b710b02c737ff2e982cf83fbfdfbb877"
    end
  end

  def install
    bin.install "hivebear"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/hivebear --version")
  end

  def caveats
    <<~EOS
      Get started with HiveBear:

        hivebear quickstart

      This will profile your hardware, recommend the best model,
      download it, and start an interactive chat session.
    EOS
  end
end
