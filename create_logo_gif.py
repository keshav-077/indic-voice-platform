"""
VoiceBharat Pro Logo GIF Generator
Creates an animated logo with Indian tricolor theme and Ashoka Chakra
"""

from PIL import Image, ImageDraw, ImageFont
import math

def create_voicebharat_logo():
    # Settings
    width, height = 800, 400
    frames = 60
    fps = 30
    
    # Indian tricolor theme
    bg_color = (255, 255, 255)
    saffron = (255, 153, 51)   # #ff9933
    white = (255, 255, 255)
    green = (19, 136, 8)       # #138808
    navy = (0, 0, 128)         # #000080
    
    images = []
    
    for frame in range(frames):
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Animation progress
        progress = frame / frames
        angle = progress * 2 * math.pi
        
        # Draw tricolor background stripes (subtle, pulsing)
        stripe_height = 60
        stripe_y = 80
        opacity_pulse = int(80 + 40 * math.sin(angle))
        
        # Saffron stripe
        stripe_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        stripe_draw = ImageDraw.Draw(stripe_img)
        stripe_draw.rectangle(
            [0, stripe_y, width, stripe_y + stripe_height],
            fill=(*saffron, opacity_pulse)
        )
        img = Image.alpha_composite(img.convert('RGBA'), stripe_img)
        img = img.convert('RGB')
        
        # Green stripe
        stripe_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        stripe_draw = ImageDraw.Draw(stripe_img)
        stripe_draw.rectangle(
            [0, stripe_y + stripe_height * 2, width, stripe_y + stripe_height * 3],
            fill=(*green, opacity_pulse)
        )
        img = Image.alpha_composite(img.convert('RGBA'), stripe_img)
        img = img.convert('RGB')
        
        # Center position
        center_x, center_y = width // 2, height // 2 - 10
        
        # Draw Ashoka Chakra inspired design
        chakra_radius = 90
        pulse = math.sin(angle) * 2
        
        # Outer circle
        draw.ellipse(
            [center_x - chakra_radius - pulse, center_y - chakra_radius - pulse,
             center_x + chakra_radius + pulse, center_y + chakra_radius + pulse],
            outline=navy,
            width=3
        )
        
        # Middle circle
        draw.ellipse(
            [center_x - chakra_radius + 15, center_y - chakra_radius + 15,
             center_x + chakra_radius - 15, center_y + chakra_radius - 15],
            outline=navy,
            width=2
        )
        
        # 24 spokes (rotating)
        spoke_rotation = progress * 360
        for i in range(24):
            spoke_angle = math.radians(i * 15 + spoke_rotation)
            inner_radius = chakra_radius - 20
            outer_radius = chakra_radius - 5
            
            x1 = center_x + inner_radius * math.cos(spoke_angle)
            y1 = center_y + inner_radius * math.sin(spoke_angle)
            x2 = center_x + outer_radius * math.cos(spoke_angle)
            y2 = center_y + outer_radius * math.sin(spoke_angle)
            
            draw.line([x1, y1, x2, y2], fill=navy, width=2)
        
        # Central hub with microphone
        hub_radius = 22
        draw.ellipse(
            [center_x - hub_radius, center_y - hub_radius,
             center_x + hub_radius, center_y + hub_radius],
            fill=navy
        )
        
        # Microphone in center (small, tricolor)
        mic_width, mic_height = 12, 20
        mic_x = center_x - mic_width // 2
        mic_y = center_y - mic_height // 2 - 5
        
        # Saffron part
        draw.rounded_rectangle(
            [mic_x, mic_y, mic_x + mic_width, mic_y + mic_height * 0.4],
            radius=6,
            fill=saffron
        )
        # White part
        draw.rectangle(
            [mic_x, mic_y + mic_height * 0.4, mic_x + mic_width, mic_y + mic_height * 0.6],
            fill=white
        )
        # Green part
        draw.rounded_rectangle(
            [mic_x, mic_y + mic_height * 0.6, mic_x + mic_width, mic_y + mic_height],
            radius=6,
            fill=green
        )
        
        # Microphone stand
        stand_x = center_x
        stand_y = mic_y + mic_height
        draw.line([stand_x - 2, stand_y, stand_x - 2, stand_y + 8], fill=white, width=1)
        draw.line([stand_x + 2, stand_y, stand_x + 2, stand_y + 8], fill=white, width=1)
        draw.line([stand_x - 6, stand_y + 8, stand_x + 6, stand_y + 8], fill=white, width=2)
        
        # Orbiting language symbols
        indian_langs = ['हि', 'த', 'বা', 'తె']
        colors = [saffron, green, navy, saffron]
        
        for i, (char, color) in enumerate(zip(indian_langs, colors)):
            orbit_angle = angle + i * (2 * math.pi / len(indian_langs))
            orbit_radius = 110
            char_x = center_x + orbit_radius * math.cos(orbit_angle)
            char_y = center_y + orbit_radius * math.sin(orbit_angle)
            
            try:
                char_font = ImageFont.truetype("arial.ttf", 24)
            except:
                char_font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), char, font=char_font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            
            draw.text(
                (char_x - char_width/2, char_y - char_height/2),
                char,
                fill=color,
                font=char_font
            )
        
        # Corner decorative pulsing dots
        corners = [
            (50, 100, saffron),
            (width - 50, 100, green),
            (50, height - 100, navy),
            (width - 50, height - 100, saffron)
        ]
        
        for i, (cx, cy, color) in enumerate(corners):
            dot_pulse = math.sin(angle + i * math.pi / 2) * 2
            dot_radius = 5 + dot_pulse
            draw.ellipse(
                [cx - dot_radius, cy - dot_radius,
                 cx + dot_radius, cy + dot_radius],
                fill=color
            )
        
        # Add text
        try:
            font_large = ImageFont.truetype("arial.ttf", 48)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Main title with tricolor
        text_y = height - 95
        
        # "Voice" in saffron
        voice_text = "Voice"
        bbox = draw.textbbox((0, 0), voice_text, font=font_large)
        voice_width = bbox[2] - bbox[0]
        
        # "Bharat" in navy
        bharat_text = "Bharat"
        bbox = draw.textbbox((0, 0), bharat_text, font=font_large)
        bharat_width = bbox[2] - bbox[0]
        
        # "Pro" in green
        pro_text = " Pro"
        bbox = draw.textbbox((0, 0), pro_text, font=font_large)
        pro_width = bbox[2] - bbox[0]
        
        total_width = voice_width + bharat_width + pro_width
        start_x = (width - total_width) // 2
        
        draw.text((start_x, text_y), voice_text, fill=saffron, font=font_large)
        draw.text((start_x + voice_width, text_y), bharat_text, fill=navy, font=font_large)
        draw.text((start_x + voice_width + bharat_width, text_y), pro_text, fill=green, font=font_large)
        
        # Subtitle
        subtitle = "22 Indian Languages • Enterprise AI"
        sub_bbox = draw.textbbox((0, 0), subtitle, font=font_small)
        sub_width = sub_bbox[2] - sub_bbox[0]
        sub_x = (width - sub_width) // 2
        draw.text((sub_x, text_y + 60), subtitle, fill=(100, 100, 100), font=font_small)
        
        images.append(img)
    
    # Save as GIF
    output_path = 'VoiceBharat_Pro_Logo.gif'
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000//fps,
        loop=0,
        optimize=False
    )
    
    print(f"✅ Logo saved as: {output_path}")
    print(f"   Size: {width}x{height}")
    print(f"   Frames: {frames}")
    print(f"   FPS: {fps}")

if __name__ == "__main__":
    create_voicebharat_logo()
