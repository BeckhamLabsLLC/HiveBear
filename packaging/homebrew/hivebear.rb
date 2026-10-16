class Hivebear < Formula
  desc "Run local AI models, with picks matched to your hardware"
  homepage "https://hivebear.com"
  version "0.1.12"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-apple-darwin.tar.gz"
      sha256 "506fc7d3e6a8267b6b93fff631613d56c8a52123172162f7a473b62f8286b8fd"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-apple-darwin.tar.gz"
      sha256 "8c140533d8be55856388e6fe11fc6eeec60d0c4e829128ef9fef83d839386ed6"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "6aeabd190073850e041cfe6cbbf612c566175e8d385d05175e82319dd924ee91"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "73fa8a5f81a88209669998e97d17f01957238d57a59750d15f8fd565ce4b0c3b"
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
