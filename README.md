# 06BinaryTree

二叉树的遍历：

递归三要素：
1. 确定递归函数的参数和返回值：List<Integer> inorderTraversal(TreeNode root)
2. 确定终止条件（一般都是root == null）：if (root == null)  return;
3. 确定单层递归的逻辑(前中序遍历想好）：inorder(root.left, list);          list.add(root.val);         inorder(root.right, list);

用stack迭代遍历：
1.前序遍历（中左右）先push(root) -> pop(root) -> push(root.right) -> push(root.left) -> pop(root.left) -> push(root.left.right) -> push(root.left.left)
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            result.add(node.val);
            if (node.right != null){
                stack.push(node.right);
            }
            if (node.left != null){
                stack.push(node.left);
            }
        }
        return result;
    }
}
2.中序遍历（左中右）：入栈顺序： 左-右

class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()){
           if (cur != null){//不管传入的是左孩子还是右孩子，都去查他的左边。
               stack.push(cur);
               cur = cur.left;//左边全部入完
           }else{
               cur = stack.pop();//左边存完了开始pop（）一个root出来，然后查root的左边。
               result.add(cur.val);
               cur = cur.right;//左边已经没有，查一下右边，进入下一层循环。
           }
        }
        return result;
    }
}
3.后序遍历（左右中）： 前序“中左右”在push的时候改为先左后右，得到“中右左”，然后reverse得到“左右中”。
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            result.add(node.val);
            if (node.left != null){
                stack.push(node.left);
            }
            if (node.right != null){
                stack.push(node.right);
            }
        }
        Collections.reverse(result);
        return result;
    }
}

层序遍历：二叉树关于以层为基础的题目都适用，可用队列先进先出，符合一层一层遍历的逻辑。
class Solution {
    public void checkFun02(TreeNode node) {
        if (node == null) return;
        Queue<TreeNode> que = new LinkedList<TreeNode>();
        que.offer(node);

        while (!que.isEmpty()) {
            List<Integer> itemList = new ArrayList<Integer>();
            int len = que.size();

            while (len > 0) {
                TreeNode tmpNode = que.poll();
                itemList.add(tmpNode.val);

                if (tmpNode.left != null) que.offer(tmpNode.left);
                if (tmpNode.right != null) que.offer(tmpNode.right);
                len--;
            }

            resList.add(itemList);
        }
    }
}

226.翻转二叉树
力扣题目链接(opens new window)

翻转一棵二叉树。

示例：
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

解题递归三要素：
1. 返回值和参数：返回值还是root，参数就是原来的数root。
2. 终止条件：遍历到root == null 的时候，就可以 return null。
3. 单层遍历逻辑，采用中序，先处理中间节点，也就是root.left 和 root.right的交换，然后再遍历他们的下面。

代码实现：
class Solution {
   /**
     * 前后序遍历都可以
     * 中序不行，因为先左孩子交换孩子，再根交换孩子（做完后，右孩子已经变成了原来的左孩子），再右孩子交换孩子（此时其实是对原来的左孩子做交换）
     */
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        invertTree(root.left);
        invertTree(root.right);
        swapChildren(root);
        return root;
    }

    private void swapChildren(TreeNode root) {
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
    }
}
也可以采用bfs, 每次queue.poll()出来一个node时候，把他们的左右互换一下，然后再执行offer。

101. 对称二叉树

给定一个二叉树，检查它是否是镜像对称的。
Input: root = [1,2,2,3,4,4,3]
Output: true

解题递归三要素：
1. 返回值和参数：返回值是boolean,参数是原来数的root.left, root.right；
2. 终止条件：要对比的左子树，右子树情况，所以两个节点要分类讨论 == null的情况。因为是比较左右子树，所以可以单独写一个helper的方法，直接传参数
   2.1、left == null && right != null, false；
   2.2、left != null && right == null，false；
   2.3、left == null && right == null, true；
   2.4、left != null && right != null && left != right, false;
3. 单层递归逻辑（左右中），有返回值的一般都是left = 遍历左边，right = 遍历右边。中：然后对left跟right进行对比、联合操作。

代码实现：
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root == null) return true;
        if( helper(root.left, root.right)){
            return true;
        }else{
            return false;
        }
    }
    public boolean helper(TreeNode left,TreeNode right){
        if(left == null && right == null) return true;
        if(left != null && right == null) return false;
        if(left == null && right != null) return false;
        if(left.val != right.val) return false;
        //left == right,所以进入下一层比较，先左右，然后中，这是后序遍历。
        boolean outside = helper(left.left, right.right);
        boolean inside = helper(left.right, right.left);
        return inside && outside;

    }
}

104.二叉树的最大深度
力扣题目链接(opens new window)

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例： 给定二叉树 [3,9,20,null,null,15,7]，
      返回它的最大深度 3。

解题递归三要素：
1. 返回值和参数：返回值int， 参数就是原来的数root。
2. 终止条件： 当root == null， 退出return 0；
3. 单层递归逻辑（左右中），有返回值的一把都是left = 遍历左边。right = 遍历右边。中：然后对left跟right进行对比、联合操作。

代码实现：
class solution {
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        return Math.max(leftDepth, rightDepth) + 1;
    }
} 

111.二叉树的最小深度
力扣题目链接(opens new window)

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

说明: 叶子节点是指没有子节点的节点。

示例:

给定二叉树 [3,9,20,null,null,15,7],
输出：最小深度 2.

解题递归三要素：
1. 返回值和参数：返回值int， 参数就是原来树的root。
2. 终止条件：如果root == null， return 0；
3. 单层递归逻辑（左右中）。有返回值的一般都要用left，right接住递归的返回值。中：再对left跟right进行对比、联合操作。这题是比小，那就是(left ,right)+1，如果root下面只有一个节点，那就说明到低层了，可以这个层深度（不为空那个节点）+1，因为null就是没有，表示可以不继续往下了，但是null返回的是0，这不合理，所以取部位null的同层节点的高度。

代码实现：
class Solution {
    /**
     * 递归法，相比求MaxDepth要复杂点
     * 因为最小深度是从根节点到最近**叶子节点**的最短路径上的节点数量
     */
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftDepth = minDepth(root.left);
        int rightDepth = minDepth(root.right);
        if (root.left == null) {
            return rightDepth + 1;
        }
        if (root.right == null) {
            return leftDepth + 1;
        }
        // 左右结点都不为null
        return Math.min(leftDepth, rightDepth) + 1;
    }
}

222.完全二叉树的节点个数
力扣题目链接(opens new window)

给出一个完全二叉树，求出该树的节点个数。

示例 1：

输入：root = [1,2,3,4,5,6]
输出：6

解题递归三要素：
1.返回值和参数:返回值是int，代表节点个数，参数就是传入的树root。
2.终止条件，当root == null 的时候，退出return 0；
3.单层递归逻辑：采用左右中，有返回值的要用left和right接住左右的遍历返回结果。中:对于左右遍历返回的节点数相加+1（root）。

代码实现;
class Solution {
    public int countNodes(TreeNode root) {
        if(root == null) return 0;
        int countLeft = countNodes(root.left);
        int countRight = countNodes(root.right);
        int count = countLeft + countRight + 1;
        return count;
    }
}

110.平衡二叉树
力扣题目链接(opens new window)

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。

示例 1:

给定二叉树 [3,9,20,null,null,15,7]
返回 true 。

解题递归三要素：
这题跟求深度相似，这道题要考虑做对比，看左右子树高度有没有超过1，有可能过程某颗子树就不平衡，也可能左子树平衡，右子树平衡，然后共同root这里发现不平衡。
1. 返回值和参数：返回值是boolean, 参数就是传入的树root。
2. 终止条件：当root == null，return 0;
3. 单层递归逻辑（左右中）， 有返回值所以需要left、right来接住左右的遍历，中：如果过层中发现了不平衡（left||right == -1），则可以return 标记高度-1了，如果没出现的话可以用差值来判断。

代码实现：
class Solution {
    public boolean isBalanced(TreeNode root) {
        return height(root) >= 0 ;
    }
    public int height(TreeNode root) {
        if(root == null) return 0;
        int left = height(root.left);
        int right = height(root.right);
        if (right < 0 || left < 0) return -1; 
        if(Math.abs(left-right) > 1) return -1;
        return Math.max(left, right) +1;
    }
}

257. 二叉树的所有路径
力扣题目链接(opens new window)

给定一个二叉树，返回所有从根节点到叶子节点的路径。

说明: 叶子节点是指没有子节点的节点。

解题递归三要素，
1. 递归参数与返回值，参数为原来的树root，存所有路径的list，以及储存所有路径的output。
2. 递归终止条件，当root.left == null && root.right == null的时候，遍历到叶子节点，可以停止遍历，生成一条路径，把这条路径转化为string，然后存到output。
3. 递归顺序：中把节点放到path里面，然后遍历一直左，再一直遍历右，
特殊点：不能总是一直加，每次找到一个新路径之后，回去再找右边节点，要删除之前存的最后一个节点。不然就会出现重复节点。

class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> output = new ArrayList<>();
        if(root == null) return output;
        List<Integer> path = new ArrayList<>();
        traverse(root, path, output);
        return output;
    }
    public void traverse(TreeNode root, List<Integer> path, List<String> output){
        path.add(root.val);
        if(root.left == null && root.right == null){
            StringBuilder sb = new StringBuilder();
            for(int i = 0; i < path.size() - 1; i++){
                sb.append(path.get(i)).append("->");
            }
            sb.append(path.get(path.size() - 1));
            output.add(sb.toString());
        }
        if(root.left != null){
            traverse(root.left, path, output);
            path.remove(path.size()-1);
        }
        if(root.right != null){
            traverse(root.right, path, output);
            path.remove(path.size()-1);
        }
    }
}

404.左叶子之和
力扣题目链接(opens new window)

计算给定二叉树的所有左叶子之和。

解题思路递归三要素：
1. 递归参数和返回值：参数是root，返回值是int(sum)
2. 递归终止条件root == null，return 0; 
3. 递归顺序，遍历左边用left接住左边的左子树和，遍历右边用right接住右边的左子树和。中：判断是不是叶子节点，再判断这个叶子节点是左节点的话就加；所有又要知道当前节点的左右，还要知道当前节点是上一个节点的左。

   代码实现：
class Solution {
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) return 0;
        int leftValue = sumOfLeftLeaves(root.left);    // 左
        int rightValue = sumOfLeftLeaves(root.right);  // 右
                                                       
        int midValue = 0;
        if (root.left != null && root.left.left == null && root.left.right == null) { 
            midValue = root.left.val;
        }
        int sum = midValue + leftValue + rightValue;  // 中
        return sum;
    }
}

513.找树左下角的值
力扣题目链接(opens new window)

给定一个二叉树，在树的最后一行找到最左边的值。

解题思路：需要知道深度，并且不断更新深度，以及深度那一排对应的第一个节点。所以需要有maxDepth，当前depth， 待更新的result。
递归三要素：
1. 递归返回值和参数：需要root，还有的depth来记录当前的深度。多于题目的方法参数，因此新建一个方法。新方法不需要return，只需要在递归过程中修改result就可以。
2. 递归终止条件: 结束递归有两种方式，一种是return，另一种是全部遍历完成。本题是全部遍历结束退出递归，如果发现叶子节点，就要判断当前层是不是最大深度，是的话更新result，并且更新最大深度。
3. 递归逻辑：前序遍历，中：当root.left == null && root.right == null的时候，操作。然后向左遍历，向右遍历。
提示：如果想用return的方式，可以不要判断root.left != null，root.right != null。这样递归会走到root == null 这里结束。
res 跟 maxDepth必须作为globle，不然不会被改变，因为integer是 pass by value 的。

代码实现：
class Solution {
    int res = 0;
    int maxDepth = Integer.MIN_VALUE;
    public int findBottomLeftValue(TreeNode root) {
        res = root.val;
        int depth = 0;
        traversal(root, depth);
        return res;
    }
    public void traversal(TreeNode root, int depth ){
        if(root == null) return;
        if(root.left == null && root.right == null){
            if(depth > maxDepth){
                maxDepth = depth;
                res = root.val;
            }
        } 
        if(root.left != null){
            depth++;
            traversal(root.left, depth);
            depth--;
        }
        if(root.right != null){
            depth++;
            traversal(root.right, depth);
            depth--;
        }
    }   
}

112. 路径总和
力扣题目链接(opens new window)

给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。

说明: 叶子节点是指没有子节点的节点。

示例: 给定如下二叉树，以及目标和 sum = 22，
返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。

解题思路：可以通过不段修改sum的值传入到新的递归中来判断。每进入下一层都传sum= sum - 上一层的root值。
递归三要素：
1. 递归返回值和参数：参数为root，和sum，返回值就是boolean.
2. 递归终止条件：当root == null 的时候，结束递归return false；
3. 递归顺序，前序遍历， 中：如果当前是叶子节点，并且root的值等于sum，return true，否则的话遍历左边用left 接住结果，遍历右边，用left接住结果。只要出现true就可以结束。

代码实现：
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root == null) return false;
        if(root.left == null && root.right == null) return root.val == targetSum;
        if(hasPathSum(root.left, targetSum - root.val)) return true;
        if(hasPathSum(root.right, targetSum - root.val)) return true;
        return false;
    }
}
