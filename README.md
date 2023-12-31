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

654.最大二叉树

给定一个不含重复元素的整数数组。一个以此数组构建的最大二叉树定义如下：

二叉树的根是数组中的最大元素。
左子树是通过数组中最大值左边部分构造出的最大二叉树。
右子树是通过数组中最大值右边部分构造出的最大二叉树。
通过给定的数组构建最大二叉树，并且输出这个树的根节点。

解题思路：
1. 返回值和参数：返回值即是root节点，参数为传入的数组，以及构建子树时要查找的新节点的index范围：leftIndex, rightIndex.
2. 递归终止条件：当leftIndex < rightIndex的时候，就是已经无法查找，可以结束递归，return null。（就是return 到上一层）；
3. 递归顺序，中左右，中：找到当前index范围内的最大值，作为一个root，并且记住当前的最大值对应的maxIndex，然后遍历左边，把maxIndex 作为新的rightIndex, 遍历右边，把maxIndex+1 作为新的leftIndex, 
提示：return是从下往上连接的，
代码实现;
class Solution {
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return constructMaximumBinaryTree(nums, 0, nums.length);
    }

    public TreeNode constructMaximumBinaryTree(int[] nums, int leftIndex, int rightIndex) {
        if(rightIndex - leftIndex < 1) return null;
        if(rightIndex - leftIndex == 1) return new TreeNode(nums[leftIndex]);
        int maxNum = nums[leftIndex];
        int maxIndex = leftIndex;
        for(int i = leftIndex + 1; i < rightIndex; i++){
            if(nums[i] > maxNum){
                maxNum = nums[i];
                maxIndex = i;
            }
        }
        TreeNode root = new TreeNode(maxNum);
        root.left = constructMaximumBinaryTree(nums, leftIndex, maxIndex);
        root.right = constructMaximumBinaryTree(nums, maxIndex + 1, rightIndex);
        return root;
    }
}

617.合并二叉树
力扣题目链接(opens new window)

给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。

你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

解题思路：同时遍历两颗树，如果这个位置节点两棵树都存在，就相加并作为新节点，如果只有一颗有，那就取这颗的，如果都是null，就return null。
1. 递归返回值和参数：返回值是新的root，参数就两棵树的root。
2. 递归终止条件：当root1 == null || root1 == null的时候，就return 不为空的root，
3. 递归顺序，中：t1的value + t2的value作为t1的value，遍历左边用left接住，遍历右边right接住，最终人突然t1;

代码实现：
class Solution {
    // 递归
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null) return root2;
        if (root2 == null) return root1;
        root1.val += root2.val;//中
        root1.left = mergeTrees(root1.left,root2.left);//左
        root1.right = mergeTrees(root1.right,root2.right);//右
        return root1;
    }
}

700.二叉搜索树中的搜索

给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。

解题思路：可以普通遍历所有节点来查询是否有root.val == val,但是因为二叉搜索树的有序性特征，可以在遍历之前做比较，val大于root.val 的话，往右走，小的话往左右。
1. 参数与返回值：返回值是null 或者找到的root.
2. 终止条件：如果root == null（找不到）或者root.val == val （找到了），return root。
3. 单层遍历顺序：中左右，中：判断root.val == val， 如果val 大于root.val向左遍历，否则向右，需要用一个left跟right来接住遍历结果，

代码实现：
class Solution {
    // 递归，利用二叉搜索树特点，优化
    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null || root.val == val) {
            return root;
        }
        if (val < root.val) {
            return searchBST(root.left, val);
        } else {
            return searchBST(root.right, val);
        }
    }
}

class Solution {
    // 递归，普通二叉树
    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null || root.val == val) {
            return root;
        }
        TreeNode left = searchBST(root.left, val);
        if (left != null) {
            return left;
        }
        return searchBST(root.right, val);
    }
}

98.验证二叉搜索树

给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。

解题思路：判断二叉树搜索是否有效，主要是判断左子树是否有效，右子树是否有效。并且结合二叉搜索树的中序遍历是一个递增的序列。所以只要检查后面的值都比前面的值小就可以了。
1. 返回值和参数：boolean 跟传入的树root。
2. 终止条件： 如果root == null， return false；
3. 中序遍历，一开始设定max为long最小值，然后遍历左边，中：root 大于 max，然后更新max的值，小于的话return false。然后遍历右边。

代码实现：
class Solution {
    // 递归
    TreeNode max;
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        // 左
        boolean left = isValidBST(root.left);
        if (!left) {
            return false;
        }
        // 中
        if (max != null && root.val <= max.val) {
            return false;
        }
        max = root;
        // 右
        boolean right = isValidBST(root.right);
        return right;
    }
}

530.二叉搜索树的最小绝对差
力扣题目链接(opens new window)

给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。

解题思路： 除了遍历所有节点外，同时需要一个min来一直更新，并且需要pre节点来跟当前节点对减。
1. 返回值和参数：minValue 是全局变量，所以遍历方法还是可以返回值为void，参数为树root，
2. 终止条件：当 root == null，就return返回。
3. 遍历顺序：左中右，左边遍历，中：用pre来记录root，然后逐个几点对比更新minValue的值，再遍历右边。

代码实现：
class Solution {
    TreeNode pre;// 记录上一个遍历的结点
    int result = Integer.MAX_VALUE;
    public int getMinimumDifference(TreeNode root) {
       if(root==null)return 0;
       traversal(root);
       return result;
    }
    public void traversal(TreeNode root){
        if(root==null)return;
        //左
        traversal(root.left);
        //中
        if(pre!=null){
            result = Math.min(result,root.val-pre.val);
        }
        pre = root;
        //右
        traversal(root.right);
    }
}

236. 二叉树的最近公共祖先
力扣题目链接(opens new window)

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

解题思路：这个属于要从下往上要找的题目，所有采用后续遍历，先去左子树，右子树找，找到p或者q那就向上返回，没找到就返回null，最后判断左右两边情况，都是null，就说明找不到，一个null，一个有，那就是有的那个，如果两边都有，那就是左右的根root。
1. 返回值和参数：返回值是节点，传入的是数root，还有待查找的p、q.
2. 终止条件，找到p或者q,或者找到叶子节点也没找到，就是return，return内容可以参考如果这棵树只有一个节点满足这些情况的话，应该要return自己本身，所以return root。
3. 单层递归逻辑，左右遍历，用left，right接住返回值，中（root）：对左右返回值节点做判断。

代码实现：
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) { // 递归结束条件
            return root;
        }
        // 后序遍历
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        if(left == null && right == null) { // 若未找到节点 p 或 q
            return null;
        }else if(left == null && right != null) { // 若找到一个节点
            return right;
        }else if(left != null && right == null) { // 若找到一个节点
            return left;
        }else { // 若找到两个节点
            return root;
        }
    }
}
