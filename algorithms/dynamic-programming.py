from manim import *


class VerticalRectangle(Scene):
    def construct(self):
        # Create a vertical rectangle
        rectangle = Rectangle(width=2, height=5)

        # Create horizontal lines to divide the rectangle into 5 rows
        for i in range(1, 5):
            line = Line(start=rectangle.get_left(), end=rectangle.get_right())
            line.move_to(rectangle.get_bottom() + i * rectangle.get_height() / 5 * UP)
            self.add(line)

        # Add the rectangle to the scene
        self.play(Create(rectangle))

        # Add the horizontal lines to the scene
        for i in range(1, 5):
            line = Line(start=rectangle.get_left(), end=rectangle.get_right())
            line.move_to(rectangle.get_bottom() + i * rectangle.get_height() / 5 * UP)
            self.play(Create(line))

        self.wait(2)
