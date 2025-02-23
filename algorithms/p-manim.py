from manim import *


class VerticalRectangle(Scene):

    def getPosition(self, row, rectangle):
        return rectangle.get_bottom() + (row + 0.5) * rectangle.get_height() / 5 * UP

    def construct(self):
        s = "test"
        lines = []
        text = Text(s)
        # self.play(Write(text), run_time=1)
        # self.play(Write(NumberPlane().add_coordinates()), run_time=1)
        self.wait(2)
        self.play(Write(text), run_time=1)
        self.play(text.animate.move_to([-3,-2, 0]), run_time=2)
        # Create a vertical rectangle
        rectangle = Rectangle(width=2, height=5)

         # Add the rectangle to the scene
        self.play(Create(rectangle))
        for i in range(1, 5):
            line = Line(start=rectangle.get_left(), end=rectangle.get_right())
            line.move_to(rectangle.get_bottom() + i * rectangle.get_height() / 5 * UP)
            lines.append(line)
            

        for line in lines:
            self.add(line)

        # Define texts and their final positions
        texts = ["fib(5, None)"]
        text_objects = []

        # for i, text in enumerate(texts):
            # Create a text object off the screen (to the left)
        text_obj = Text(texts[0], font_size=20)
        
        text_obj.move_to(rectangle.get_left() + LEFT * 3)  # Start outside the rectangle

        text_objects.append(text_obj)  # Store text objects for later use
        self.add(text_obj)

        # Calculate the final position for the text
        final_position = self.getPosition(0, rectangle)

        # Animate the movement of the text from outside into the rectangle
        self.wait(3)
        self.play(text_obj.animate.move_to(final_position), run_time=1)

    
        self.wait(2)
        
        self.play(FadeOut(rectangle), 
                  *[FadeOut(line) for line in lines], 
                  FadeOut(text_objects[0]),
                  FadeOut(text))
        
        self.wait(3)

        ####################################################

        text = Text("a b c", font_size=72)
        self.play(Write(text))
        
        # Calculate the position of the letter "a"
        letter_a = text[0]  # Get the first character, which is "a"
        
        # Create an underline using a Line
        underline = Line(
            color='red',
            start=letter_a.get_left() + LEFT * 0.1,  # Start slightly left of 'a'
            end=letter_a.get_right() + RIGHT * 0.1,  # End slightly right of 'a'
            stroke_width=5  # Thickness of the underline
        )
        
        # Position the line below the letter "a"
        underline.move_to(letter_a.get_bottom() + DOWN * 0.1)  # Move it below 'a'
        
        # Animate the underline appearing
        
        self.play(Create(underline))
        self.wait(2)
        self.play(FadeOut(underline), FadeOut(text))
        ####################################
        # block_of_text = MarkupText(
        #     "def fibo(n, mem = None):\n"
        #     "     if mem is None:\n"
        #     "       mem = {}"
        #     # "     if n <= 1:\n"
        #     # "       return n\n"
        #     # "   mem[n] = (fibo(n - 1, mem) + fibo(n - 2, mem))\n"
        #     "   return mem[n]",
        #     font_size=36,
        #     line_spacing=0.75
        #     )
        
        code_snippet = Text(
            "def fibo(n, mem = None):\n"

            "    if mem is None:\n"
            "        mem = {}\n"

            "    if n <= 1:\n"
            "       return n\n"

            "    if n in mem:\n"
            "       return mem[n]\n"

            "    mem[n] = fibo(n - 1, mem) + fibo(n - 2, mem)\n"
            "    return mem[n]\n",
            # language="Python",  # Set the programming language for syntax highlighting
            font_size=24,       # Adjust font size
            line_spacing=0.9,   # Adjust line spacing
            # background=None, # Optional: adds a background
            # style='colorful'     # Syntax highlighting style
        )

        code_snippet.move_to(ORIGIN)

        # Animate the writing of the text on screen
        self.play(Write(code_snippet))

        underline = Line(
            color='red',
            start= LEFT * 3 + DOWN * 1.7,   # Adjust to start according to your dimension
            end=RIGHT * 3.2 + DOWN * 1.7,   # Adjust to end according to your dimension
            stroke_width=2,
        )


        self.play(Create(underline))
        
        # Wait a moment before ending the scene
        self.wait(3)

        self.play(FadeOut(code_snippet), FadeOut(underline))

        ################################



         # Create two binary trees
        tree1_root = self.create_binary_tree("4", "3", "2", ORIGIN + LEFT * 3)
        tree2_root = self.create_binary_tree("3","2", "1", ORIGIN + RIGHT * 3)
        
        

        pos = ORIGIN + UP * 2
        rooot = Circle(radius=0.3).move_to(pos)
        root_label = Text("5").scale(0.5).move_to(pos)

        # Animate creating the root node
        self.add(rooot)
        self.add(root_label)

        # Define children position offsets
        left_position = pos + LEFT * 3 + DOWN * 2
        right_position = pos + RIGHT * 3 + DOWN * 2

        self.add(self.get_line_pos(pos, left_position))
        self.add(self.get_line_pos(pos, right_position))
       
        self.wait(2)

        # self.play(Create(self.get_line_pos(pos, left_position)))
        # self.play(Create(self.get_line_pos(pos, right_position)))

    def get_line_pos(self, root_pos, node_pos):
        return Line(root_pos + DOWN * 0.3, node_pos + UP * 0.3)

    def create_binary_tree(self, root_value, left_val, right_val, position):
        # Create a circle for the root node
        root_node = Circle(radius=0.3).move_to(position)
        root_label = Text(root_value).scale(0.5).move_to(position)

        # Animate creating the root node
        self.add(root_node) 
        self.add(root_label)

        # Define children position offsets
        left_position = position + LEFT * 1 + DOWN * 1
        right_position = position + RIGHT * 1 + DOWN * 1

        # Draw left child (C)
        left_node = Circle(radius=0.3).move_to(left_position)
        left_label = Text(left_val).scale(0.5).move_to(left_position)

        self.add(Line(position + DOWN * 0.3, left_position + UP * 0.3))  # Start from above the root
        self.add(left_node)
        self.add(left_label)

        # Draw right child (D)
        right_node = Circle(radius=0.3).move_to(right_position)
        right_label = Text(right_val).scale(0.5).move_to(right_position)

        self.add(Line(position + DOWN * 0.3, right_position + UP * 0.3))
        self.add(right_node)
        self.add(right_label)

        # Return the root node to complete the tree drawing
        return root_node

